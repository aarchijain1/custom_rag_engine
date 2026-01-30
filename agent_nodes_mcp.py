"""
Agent Nodes for LangGraph-based RAG system
LLM-based routing (Gemini) + ChromaDB retrieval
Uses HTTP-based MCP client instead of stdio
"""

from typing import TypedDict, List, Dict, Any
from mcp_client import VectorStoreMCP  # NEW: HTTP-based MCP client
from config import (
    CLASSIFICATION_PROMPT, 
    ENABLE_RAG_CLASSIFICATION,
    MCP_SERVER_URL,
    MCP_AUTO_START
)
from llm import llm

# -------------------------------------------------------------------
# STATE
# -------------------------------------------------------------------

class AgentState(TypedDict):
    question: str
    retrieved_docs: List[Dict[str, Any]]
    final_answer: str
    use_rag: bool
    error: str

# -------------------------------------------------------------------
# VECTOR STORE (singleton) - NOW USES HTTP MCP
# -------------------------------------------------------------------

# Initialize with HTTP transport and auto-start capability
vector_store = VectorStoreMCP(
    server_url=MCP_SERVER_URL,
    auto_start_server=MCP_AUTO_START
)

# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def create_initial_state(question: str) -> AgentState:
    return {
        "question": question,
        "retrieved_docs": [],
        "final_answer": "",
        "use_rag": False,
        "error": "",
    }

def get_vector_store_stats() -> dict:
    """Get vector store statistics (triggers MCP server start if needed)"""
    return vector_store.stats()

# -------------------------------------------------------------------
# MASTER ROUTER (GEMINI)
# -------------------------------------------------------------------

def master_agent_node(state: AgentState) -> AgentState:
    """
    Master agent that decides whether to use RAG or direct answer
    Pure Python logic - NO MCP involved here
    """
    if not ENABLE_RAG_CLASSIFICATION:
        state["use_rag"] = True
        return state

    response = llm.invoke(
        CLASSIFICATION_PROMPT.format(question=state["question"])
    )

    decision = response.content.strip().lower()
    state["use_rag"] = (decision != "direct")  # Fail-safe → RAG
    return state

# -------------------------------------------------------------------
# ROUTING
# -------------------------------------------------------------------

def route_after_master(state: AgentState) -> str:
    """Pure Python routing logic - NO MCP"""
    return "retrieve" if state["use_rag"] else "answer_direct"

# -------------------------------------------------------------------
# RETRIEVAL (Uses MCP for vector store access)
# -------------------------------------------------------------------

def retrieval_node(state: AgentState) -> AgentState:
    """
    Retrieval node - ONLY place where MCP is used
    MCP server starts automatically on first call if auto_start=True
    """
    try:
        # This will trigger MCP server start if needed
        state["retrieved_docs"] = vector_store.search(state["question"])
    except Exception as e:
        state["error"] = str(e)
        state["retrieved_docs"] = []

    return state

def route_after_retrieval(state: AgentState) -> str:
    """Pure Python routing logic - NO MCP"""
    return "answer_with_rag" if state["retrieved_docs"] else "answer_direct"

# -------------------------------------------------------------------
# ANSWER WITH RAG (NO SOURCES)
# -------------------------------------------------------------------

def answer_with_rag_node(state: AgentState) -> AgentState:
    """
    Answer using retrieved documents
    Pure Python + LLM - NO MCP involved
    """
    docs = state["retrieved_docs"]

    if not docs:
        state["final_answer"] = "No relevant documents found."
        return state

    context_blocks = [doc.get("content", "").strip() for doc in docs if doc.get("content")]
    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a Toyota Financial Services expert assistant.

Use ONLY the retrieved content to answer
3. Summarize the information in a user-friendly and concise way

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
- Include ALL unique public URLs (https://...) related to the user's query
that appear in the retrieved results
- Do NOT summarize, shorten, or omit URLs
- One URL per line
- Do NOT include URLs that are not directly related to the question
- If no valid public URLs exist, omit the Sources section entirely
    

Context:
{context}

Question:
{state["question"]}

Answer:
"""

    response = llm.invoke(prompt)
    state["final_answer"] = response.content.strip()
    return state

# -------------------------------------------------------------------
# DIRECT ANSWER
# -------------------------------------------------------------------

def answer_direct_node(state: AgentState) -> AgentState:
    """
    Direct answer without RAG
    Pure Python + LLM - NO MCP involved
    """
    response = llm.invoke(state["question"])
    state["final_answer"] = response.content.strip()
    return state