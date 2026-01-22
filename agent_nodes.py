"""
Agent Nodes for LangGraph-based RAG system
LLM-based routing (Gemini) + ChromaDB retrieval
CLICKABLE URL SOURCES
"""

from typing import TypedDict, List, Dict, Any
from vector_store import VectorStore
from config import CLASSIFICATION_PROMPT, ENABLE_RAG_CLASSIFICATION
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
# VECTOR STORE (singleton)
# -------------------------------------------------------------------

vector_store = VectorStore()


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
    return vector_store.get_stats()


# -------------------------------------------------------------------
# MASTER ROUTER (GEMINI)
# -------------------------------------------------------------------

def master_agent_node(state: AgentState) -> AgentState:
    if not ENABLE_RAG_CLASSIFICATION:
        state["use_rag"] = True
        return state

    response = llm.invoke(
        CLASSIFICATION_PROMPT.format(question=state["question"])
    )

    decision = response.content.strip().lower()

    # Fail-safe → RAG
    state["use_rag"] = (decision != "direct")
    return state


# -------------------------------------------------------------------
# ROUTING
# -------------------------------------------------------------------

def route_after_master(state: AgentState) -> str:
    return "retrieve" if state["use_rag"] else "answer_direct"


# -------------------------------------------------------------------
# RETRIEVAL
# -------------------------------------------------------------------

def retrieval_node(state: AgentState) -> AgentState:
    try:
        state["retrieved_docs"] = vector_store.query(state["question"])
    except Exception as e:
        state["error"] = str(e)
        state["retrieved_docs"] = []

    return state


def route_after_retrieval(state: AgentState) -> str:
    return "answer_with_rag" if state["retrieved_docs"] else "answer_direct"


# -------------------------------------------------------------------
# ANSWER WITH RAG (CLICKABLE URLs)
# -------------------------------------------------------------------

def answer_with_rag_node(state: AgentState) -> AgentState:
    """
    Uses Gemini to synthesize a clean answer from retrieved documents
    and appends clickable source URLs.
    """

    docs = state["retrieved_docs"]

    if not docs:
        state["final_answer"] = "No relevant documents found."
        return state

    # Build context for Gemini
    context_blocks = []
    urls = set()

    for doc in docs:
        content = doc.get("content", "").strip()
        meta = doc.get("metadata", {})

        if content:
            context_blocks.append(content)

        if meta.get("url"):
            urls.add(meta["url"])

    context = "\n\n".join(context_blocks)

    # Ask Gemini to summarize
    prompt = f"""
You are answering based ONLY on the following Toyota Financial Services documents.

Cover all the important details in your answer.
Do NOT repeat text.
Do NOT mention page numbers.
Do NOT invent information.

Context:
{context}

Question:
{state["question"]}

Answer:
"""

    response = llm.invoke(prompt)

    answer = response.content.strip()

    # Append clickable sources
    if urls:
        answer += "\n\nSources:\n" + "\n".join(f"- {url}" for url in sorted(urls))

    state["final_answer"] = answer
    return state



# -------------------------------------------------------------------
# DIRECT ANSWER
# -------------------------------------------------------------------

def answer_direct_node(state: AgentState) -> AgentState:
    state["final_answer"] = state["question"]
    return state
