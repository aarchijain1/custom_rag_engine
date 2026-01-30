"""
FAQ Sub-Agent
Handles general Toyota Financial Services questions using RAG
Updated to use HTTP MCP client
"""

from typing import Literal
from agents.shared_state import AgentState
from mcp_client import VectorStoreMCP 
from llm import llm
from config import N_RETRIEVAL_RESULTS, MCP_SERVER_URL, MCP_AUTO_START


FAQ_RAG_PROMPT = """
You are a Toyota Financial Services FAQ expert assistant.

Use ONLY the retrieved content to answer.

────────────────────
SCOPE & BEHAVIOR RULES
────────────────────
- You can answer ONLY questions related to Toyota Financial Services (TFS)
- If the question is generic, vague, or unrelated:
  → Respond: "I am a TFS assistant and can only answer questions related to Toyota Financial Services."

- If the question is a greeting:
  → Respond: "Hi, I'm the Toyota Financial Services virtual assistant."

- If no relevant information is found:
  → Say the query is out of scope or not available in TFS information

────────────────────
ANSWER STYLE
────────────────────
- Keep responses CLEAR and CONCISE
- Cover all important points, avoid repetition
- Use bullet points for lists
- Do NOT hallucinate or add extra details

────────────────────
STRICT RULES
────────────────────
- Do NOT answer from general knowledge
- Do NOT mention documents, PDFs, filenames, or internal systems
- Do NOT guess or infer beyond retrieved content

────────────────────
SOURCES (MANDATORY)
────────────────────
At the end of every valid answer:

- Add heading: "Sources"
- Include ALL unique public URLs (https://...) from retrieved results
- One URL per line
- If no valid public URLs exist, omit Sources section

Context:
{context}

Question:
{question}

Answer:
"""


class FAQAgent:
    """
    FAQ Sub-Agent
    
    Responsibilities:
    - Search knowledge base via MCP (HTTP transport)
    - Synthesize retrieved documents
    - Provide clear, concise answers
    - Include source citations
    """
    
    def __init__(self):
        self.name = "faq"
        # Initialize with HTTP MCP client (auto-starts server if needed)
        self.vector_store = VectorStoreMCP(
            server_url=MCP_SERVER_URL,
            auto_start_server=MCP_AUTO_START
        )
    
    def retrieve_documents(self, state: AgentState) -> AgentState:
        """
        Retrieve relevant documents via MCP (HTTP transport)
        MCP server starts automatically on first call if auto_start=True
        """
        try:
            docs = self.vector_store.search(
                state["question"], 
                k=N_RETRIEVAL_RESULTS
            )
            state["retrieved_docs"] = docs
            state["use_rag"] = True
            
            # Build context from retrieved docs
            context_blocks = [
                doc.get("content", "").strip() 
                for doc in docs 
                if doc.get("content")
            ]
            state["context"] = "\n\n".join(context_blocks)
            
        except Exception as e:
            state["error"] = f"Retrieval error: {str(e)}"
            state["retrieved_docs"] = []
            state["context"] = ""
        
        state["agent_history"].append("faq_retrieve")
        return state
    
    def generate_answer(self, state: AgentState) -> AgentState:
        """
        Generate answer using retrieved context
        Pure Python + LLM - NO MCP involved here
        """
        if not state["context"]:
            state["final_answer"] = "No relevant information found in the knowledge base."
            state["agent_history"].append("faq_answer")
            return state
        
        # Generate answer with context
        prompt = FAQ_RAG_PROMPT.format(
            context=state["context"],
            question=state["question"]
        )
        
        response = llm.invoke(prompt)
        state["final_answer"] = response.content.strip()
        state["agent_history"].append("faq_answer")
        
        return state
    
    def process(self, state: AgentState) -> AgentState:
        """
        Full FAQ agent workflow: retrieve → answer
        MCP only used in retrieve step
        """
        state = self.retrieve_documents(state)  # Uses MCP
        state = self.generate_answer(state)      # No MCP
        return state


# Node function for LangGraph
def faq_agent_node(state: AgentState) -> AgentState:
    """
    FAQ agent node for LangGraph
    """
    faq_agent = FAQAgent()
    return faq_agent.process(state)