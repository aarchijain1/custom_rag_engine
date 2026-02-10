"""
Claude SDK-based FAQ Agent
Uses Anthropic's Claude Agent SDK with MCP integration for RAG
"""

from typing import Any, Dict, List
import asyncio
import sys
import os

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    tool,
    create_sdk_mcp_server,
    AssistantMessage,
    TextBlock,
)
from anthropic import Anthropic

from agents.shared_state import AgentState
from mcp_client import VectorStoreMCP, UnifiedMCPClient
from config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    CLAUDE_TEMPERATURE,
    CLAUDE_MAX_TOKENS,
    N_RETRIEVAL_RESULTS,
    MCP_SERVER_URL,
    MCP_AUTO_START
)


class ClaudeFAQAgent:
    """Claude Agent SDK-based FAQ Agent with strict RAG grounding"""

    def __init__(self):
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not found in .env file")

        # Persistent MCP clients (avoid reconnect on every tool call)
        self.vector_store = VectorStoreMCP(
            server_url=MCP_SERVER_URL,
            auto_start_server=MCP_AUTO_START
        )
        self.async_mcp_client = UnifiedMCPClient(MCP_SERVER_URL, MCP_AUTO_START)

        # Register MCP tool
        self.mcp_server = create_sdk_mcp_server(
            name="faq-tools",
            version="1.0.0",
            tools=[self._search_knowledge_base_tool]
        )

        # 🔒 SYSTEM PROMPT WITH STRICT GROUNDING
        system_prompt = """
You are a Toyota Financial Services (TFS) expert assistant.

You MUST ONLY answer using the provided context.

────────────────────
    SCOPE & BEHAVIOR RULES
    ────────────────────
    - You can answer ONLY questions related to Toyota Financial Services (TFS)
    - If the question is generic (e.g., "what is a computer?"), vague, or unrelated:
      → Respond exactly with:
      "I am a TFS assistant and can only answer questions related to Toyota Financial Services."

    - If the question is a greeting (e.g., "hi", "what is your name?"):
      → Respond briefly:
      "Hi, I'm the Toyota Financial Services virtual assistant."

    - If no relevant information is found in the knowledge base:
      → Say the query is out of scope or not available in TFS information

    ────────────────────
    ANSWER STYLE
    ────────────────────
    - Keep responses CLEAR, and CONCISE
    - Cover all important points, but avoid repetition
    - Use bullet points for lists
    - Do NOT hallucinate or add extra details

    ────────────────────
    STRICT RULES
    ────────────────────
    - Do NOT answer from general knowledge
    - Do NOT mention documents, PDFs, filenames, or internal systems
    - Do NOT guess or infer beyond retrieved content

     ────────────────────
    SOURCES (MANDATORY & COMPLETE)
    ────────────────────
    At the end of every valid answer:

    - Add heading exactly: Sources
    - Include ALL unique public URLs (https://...) related to the user’s query
    that appear in the retrieved results
    - **VERIFY RELEVANCE**: Only include URLs that specifically contain the answer you provided.
    - **FILTER STRICTLY**: Do NOT include generic homepages or unrelated document links.
    - One URL per line
    - Do NOT summarize, shorten, or omit URLs
    - If no *relevant* public URLs exist, omit the Sources section entirely.


"""

        self.options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            mcp_servers={"faq-tools": self.mcp_server},
            allowed_tools=["mcp__faq-tools__search_knowledge_base"]
        )

        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = CLAUDE_MODEL
        self.system_prompt = system_prompt

    # ================= TOOL =================
    @tool("search_knowledge_base", "Search the TFS knowledge base", {"query": str, "k": int})
    async def _search_knowledge_base_tool(self, args: dict[str, Any]) -> dict[str, Any]:
        query = args["query"]
        k = args.get("k", N_RETRIEVAL_RESULTS)

        try:
            docs = await self.async_mcp_client.search(query, k)

            sources, context_parts = [], []
            for doc in docs:
                if doc.get("content"):
                    context_parts.append(doc["content"].strip())

                metadata = doc.get("metadata", {})
                if metadata.get("source"):
                    sources.append(metadata["source"])
                elif metadata.get("url"):
                    sources.append(metadata["url"])

            context = "\n\n".join(context_parts)

            return {
                "content": [{
                    "type": "text",
                    "text": f"Retrieved {len(docs)} documents.\n\n{context}"
                }],
                "isError": False,
                "data": {
                    "success": True,
                    "context": context,
                    "sources": list(set(sources)),
                    "documents_found": len(docs),
                }
            }

        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Search error: {str(e)}"}],
                "isError": True,
                "data": {"success": False, "documents_found": 0, "sources": []}
            }

    # ================= MAIN AGENT FLOW =================
    async def process_question(self, question: str) -> Dict[str, Any]:
        try:
            # Fetch context directly to avoid tool-call failures
            docs = await self.async_mcp_client.search(question, N_RETRIEVAL_RESULTS)

            sources, context_parts = [], []
            for doc in docs:
                if doc.get("content"):
                    context_parts.append(doc["content"].strip())

                metadata = doc.get("metadata", {})
                if metadata.get("source"):
                    sources.append(metadata["source"])
                elif metadata.get("url"):
                    sources.append(metadata["url"])

            context = "\n\n".join(context_parts)

            # 🚨 HARD GUARDRAIL
            if not context:
                return {
                    "success": False,
                    "answer": "I'm unable to access the Toyota Financial Services knowledge base right now. Please try again later or contact Toyota Financial Services directly.",
                    "sources": [],
                    "documents_found": 0,
                    "context_used": False,
                    "model_used": self.model,
                    "search_query": question,
                    "error": "No KB context retrieved"
                }

            user_prompt = f"""Context:
{context}

User Question: {question}
"""

            response = self.client.messages.create(
                model=self.model,
                max_tokens=CLAUDE_MAX_TOKENS,
                temperature=CLAUDE_TEMPERATURE,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )

            answer_parts = []
            for block in response.content:
                if hasattr(block, "text") and block.text:
                    answer_parts.append(block.text)

            answer = "\n".join(answer_parts).strip() if answer_parts else ""

            if not answer:
                return {
                    "success": False,
                    "answer": "I'm unable to access the Toyota Financial Services knowledge base right now. Please try again later or contact Toyota Financial Services directly.",
                    "sources": [],
                    "documents_found": len(docs),
                    "context_used": True,
                    "model_used": self.model,
                    "search_query": question,
                    "error": "Empty model response"
                }

            return {
                "success": True,
                "answer": answer,
                "sources": list(set(sources)),
                "documents_found": len(docs),
                "context_used": True,
                "model_used": self.model,
                "search_query": question
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "answer": "System error occurred while processing your request.",
                "sources": [],
                "documents_found": 0
            }

    def process_question_sync(self, question: str) -> Dict[str, Any]:
        return asyncio.run(self.process_question(question))

    # ================= LANGGRAPH SUPPORT =================
    def process(self, state: AgentState) -> AgentState:
        result = self.process_question_sync(state["question"])
        state["final_answer"] = result["answer"]
        state["retrieved_docs"] = []
        state["context"] = ""
        state["agent_history"].append("claude_faq")
        return state


def claude_faq_agent_node(state: AgentState) -> AgentState:
    return ClaudeFAQAgent().process(state)


async def ask_claude_faq(question: str) -> Dict[str, Any]:
    return await ClaudeFAQAgent().process_question(question)


def ask_claude_faq_sync(question: str) -> Dict[str, Any]:
    return ClaudeFAQAgent().process_question_sync(question)
