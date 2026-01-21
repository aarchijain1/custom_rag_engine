"""
Agent Nodes for LangGraph-based RAG system
NO LangChain message objects
PURE Python + LangGraph
"""

from typing import TypedDict, List, Dict, Any
from vector_store import VectorStore


# -------------------------------------------------------------------
# State Schema (SINGLE SOURCE OF TRUTH)
# -------------------------------------------------------------------

class AgentState(TypedDict):
    question: str
    retrieved_docs: List[Dict[str, Any]]
    final_answer: str
    use_rag: bool
    error: str


# -------------------------------------------------------------------
# Vector Store (singleton-style)
# -------------------------------------------------------------------

vector_store = VectorStore()


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def create_initial_state(question: str) -> AgentState:
    return {
        "question": question,
        "retrieved_docs": [],
        "final_answer": "",
        "use_rag": False,
        "error": ""
    }


def get_vector_store_stats() -> dict:
    return vector_store.get_stats()


# -------------------------------------------------------------------
# Master Agent Node
# -------------------------------------------------------------------

def master_agent_node(state: AgentState) -> AgentState:
    """
    Decide whether to use RAG or answer directly.
    Simple heuristic to keep it deterministic.
    """
    question = state["question"].lower()

    rag_keywords = ["document", "file", "policy", "pdf", "data", "report"]

    use_rag = any(word in question for word in rag_keywords)

    state["use_rag"] = use_rag
    return state


# -------------------------------------------------------------------
# Routing after master
# -------------------------------------------------------------------

def route_after_master(state: AgentState) -> str:
    if state["use_rag"]:
        return "retrieve"
    return "answer_direct"


# -------------------------------------------------------------------
# Retrieval Node
# -------------------------------------------------------------------

def retrieval_node(state: AgentState) -> AgentState:
    try:
        docs = vector_store.search(state["question"], k=3)
        state["retrieved_docs"] = docs
    except Exception as e:
        state["error"] = str(e)
        state["retrieved_docs"] = []

    return state


# -------------------------------------------------------------------
# Routing after retrieval
# -------------------------------------------------------------------

def route_after_retrieval(state: AgentState) -> str:
    if state["retrieved_docs"]:
        return "answer_with_rag"
    return "answer_direct"


# -------------------------------------------------------------------
# Answer Nodes
# -------------------------------------------------------------------

def answer_with_rag_node(state: AgentState) -> AgentState:
    """
    Placeholder LLM logic.
    Replace with Claude / Gemini later.
    """
    docs = state["retrieved_docs"]
    context = "\n".join(doc.get("content", "") for doc in docs)

    state["final_answer"] = (
        "Answer generated using retrieved documents:\n\n" + context[:500]
    )
    return state


def answer_direct_node(state: AgentState) -> AgentState:
    state["final_answer"] = (
        "Direct answer (no documents used) for question:\n"
        + state["question"]
    )
    return state
