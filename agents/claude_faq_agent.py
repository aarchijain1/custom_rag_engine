"""
Claude SDK-based FAQ Agent
Uses Anthropic's Claude SDK with MCP integration for RAG
"""

from typing import List, Dict, Any, Optional
import asyncio
from anthropic import Anthropic
from agents.shared_state import AgentState
from mcp_client import VectorStoreMCP
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
    """
    Claude SDK-based FAQ Agent
    
    Features:
    - Uses Anthropic Claude SDK for LLM interactions
    - Integrates with MCP for document retrieval
    - Structured agent framework with tools
    - Real-time streaming responses
    """
    
    def __init__(self):
        """Initialize Claude SDK agent"""
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not found in .env file")
        
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = CLAUDE_MODEL
        self.temperature = CLAUDE_TEMPERATURE
        self.max_tokens = CLAUDE_MAX_TOKENS
        
        # Initialize VectorStoreMCP client like the original faq_agent
        self.vector_store = VectorStoreMCP(
            server_url=MCP_SERVER_URL,
            auto_start_server=MCP_AUTO_START
        )
        
        # Define agent tools
        self.tools = [
            {
                "name": "search_knowledge_base",
                "description": "Search the Toyota Financial Services knowledge base for relevant information",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant documents"
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results to retrieve (default: 3)",
                            "default": N_RETRIEVAL_RESULTS
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "answer_faq_question",
                "description": "Generate a comprehensive answer based on retrieved knowledge base information",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The user's original question"
                        },
                        "context": {
                            "type": "string",
                            "description": "Retrieved context from the knowledge base"
                        },
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of source URLs from retrieved documents"
                        }
                    },
                    "required": ["question", "context"]
                }
            }
        ]
        
        # System prompt for Claude
        self.system_prompt = """You are a Toyota Financial Services (TFS) expert assistant using the Claude AI model.

Your role is to provide accurate, helpful information about TFS policies, procedures, and services.

IMPORTANT GUIDELINES:
1. Only answer questions related to Toyota Financial Services
2. Use the provided tools to search the knowledge base before answering
3. Always cite your sources when available
4. If information is not found in the knowledge base, clearly state this
5. Be professional, clear, and concise
6. Do not make up information or speculate beyond the retrieved content

When a user asks a question:
1. First use the search_knowledge_base tool to find relevant information
2. Then use the answer_faq_question tool to provide a comprehensive answer
3. Always include source citations when available

For greetings or general conversation, respond naturally as a TFS assistant."""

    def search_knowledge_base(self, query: str, k: int = N_RETRIEVAL_RESULTS) -> Dict[str, Any]:
        """Search knowledge base using MCP - same approach as original faq_agent"""
        try:
            # Use VectorStoreMCP search directly like the original faq_agent
            docs = self.vector_store.search(query, k)
            
            # Extract sources and context
            sources = []
            context_parts = []
            
            for doc in docs:
                if doc.get("content"):
                    context_parts.append(doc.get("content", "").strip())
                
                # Extract source URLs if available
                metadata = doc.get("metadata", {})
                if "source" in metadata and metadata["source"]:
                    sources.append(metadata["source"])
                elif "url" in metadata and metadata["url"]:
                    sources.append(metadata["url"])
            
            context = "\n\n".join(context_parts) if context_parts else ""
            
            return {
                "success": True,
                "context": context,
                "sources": list(set(sources)),  # Remove duplicates
                "documents_found": len(docs),
                "query": query
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "context": "",
                "sources": [],
                "documents_found": 0,
                "query": query
            }

    def answer_faq_question(self, question: str, context: str, sources: List[str] = None) -> Dict[str, Any]:
        """Generate answer using Claude SDK (synchronous)"""
        try:
            # Prepare the prompt with context
            user_prompt = f"""User Question: {question}

Relevant Information from Knowledge Base:
{context if context else "No specific information found in the knowledge base."}

Please provide a clear, accurate answer based on the information above. If the information doesn't fully address the question, clearly state what is and isn't covered.

{f'Sources: {", ".join(sources)}' if sources else ''}"""

            # Create the message (synchronous)
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            answer = message.content[0].text
            
            return {
                "success": True,
                "answer": answer,
                "sources": sources or [],
                "context_used": bool(context),
                "model": self.model
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "answer": f"I apologize, but I encountered an error while generating your answer: {str(e)}",
                "sources": [],
                "context_used": False
            }

    def process_question(self, question: str) -> Dict[str, Any]:
        """Complete FAQ processing pipeline (synchronous)"""
        try:
            # Step 1: Search knowledge base (synchronous)
            search_result = self.search_knowledge_base(question)
            
            if not search_result["success"]:
                return {
                    "success": False,
                    "error": f"Search failed: {search_result['error']}",
                    "answer": "I'm having trouble accessing the knowledge base right now. Please try again later."
                }
            
            # Step 2: Generate answer (synchronous)
            answer_result = self.answer_faq_question(
                question=question,
                context=search_result["context"],
                sources=search_result["sources"]
            )
            
            # Combine results
            return {
                "success": answer_result["success"],
                "answer": answer_result["answer"],
                "sources": answer_result["sources"],
                "documents_found": search_result["documents_found"],
                "context_used": answer_result["context_used"],
                "model_used": answer_result.get("model", self.model),
                "search_query": search_result["query"],
                "error": answer_result.get("error")
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "answer": f"I apologize, but I encountered an unexpected error: {str(e)}",
                "sources": [],
                "documents_found": 0
            }

    def process_question_sync(self, question: str) -> Dict[str, Any]:
        """Synchronous wrapper for process_question (now just direct call)"""
        return self.process_question(question)

    # LangGraph compatibility
    def process(self, state: AgentState) -> AgentState:
        """Process using LangGraph AgentState format"""
        result = self.process_question_sync(state["question"])
        
        if result["success"]:
            state["final_answer"] = result["answer"]
            state["retrieved_docs"] = [{"content": result.get("context", "")}]
            state["context"] = result.get("context", "")
        else:
            state["final_answer"] = result["answer"]
            state["error"] = result.get("error")
            state["retrieved_docs"] = []
            state["context"] = ""
        
        state["agent_history"].append("claude_faq")
        return state


# LangGraph node function
def claude_faq_agent_node(state: AgentState) -> AgentState:
    """Claude FAQ agent node for LangGraph"""
    agent = ClaudeFAQAgent()
    return agent.process(state)


# Convenience function for direct usage
async def ask_claude_faq(question: str) -> Dict[str, Any]:
    """Convenience function to ask Claude FAQ agent a question"""
    agent = ClaudeFAQAgent()
    return await agent.process_question(question)


def ask_claude_faq_sync(question: str) -> Dict[str, Any]:
    """Synchronous convenience function"""
    agent = ClaudeFAQAgent()
    return agent.process_question_sync(question)
