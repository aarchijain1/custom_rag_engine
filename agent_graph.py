"""
LangGraph workflow definition
FINAL STABLE VERSION
"""

from langgraph.graph import StateGraph, END
from agent_nodes import (
    AgentState,
    master_agent_node,
    retrieval_node,
    answer_with_rag_node,
    answer_direct_node,
    route_after_master,
    route_after_retrieval,
)


# -------------------------------------------------------------------
# Graph Builder
# -------------------------------------------------------------------

def create_rag_agent_graph():
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("master", master_agent_node)
    workflow.add_node("retrieve", retrieval_node)
    workflow.add_node("answer_with_rag", answer_with_rag_node)
    workflow.add_node("answer_direct", answer_direct_node)

    # Entry
    workflow.set_entry_point("master")

    # Conditional routing
    workflow.add_conditional_edges(
        "master",
        route_after_master,
        {
            "retrieve": "retrieve",
            "answer_direct": "answer_direct",
        },
    )

    workflow.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "answer_with_rag": "answer_with_rag",
            "answer_direct": "answer_direct",
        },
    )

    # Terminal edges
    workflow.add_edge("answer_with_rag", END)
    workflow.add_edge("answer_direct", END)

    return workflow.compile()


# -------------------------------------------------------------------
# High-level Agent Wrapper
# -------------------------------------------------------------------

class RAGAgent:
    def __init__(self):
        print("Initializing RAG Agent...")
        self.graph = create_rag_agent_graph()
        print("✓ RAG Agent ready")

    def query(self, user_input: str) -> dict:
        from agent_nodes import create_initial_state

        state = create_initial_state(user_input)
        result = self.graph.invoke(state)

        return {
            "answer": result["final_answer"],
            "used_rag": result["use_rag"],
            "error": result["error"],
        }

    def chat(self, user_input: str) -> str:
        return self.query(user_input)["answer"]

    def get_stats(self):
        from agent_nodes import get_vector_store_stats
        return get_vector_store_stats()
