"""
Shared Agent State
Used by all agents in the system
"""

from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict):
    """
    State shared across all agents in the workflow
    """
    # User input
    question: str
    
    # Routing
    next_agent: Optional[str]  # Which agent to call next
    
    # Context and results
    retrieved_docs: List[Dict[str, Any]]
    context: str
    
    # Output
    final_answer: str
    
    # Metadata
    agent_history: List[str]  # Track which agents were called
    use_rag: bool
    error: Optional[str]


def create_initial_state(question: str) -> AgentState:
    """Create initial state for a new query"""
    return {
        "question": question,
        "next_agent": None,
        "retrieved_docs": [],
        "context": "",
        "final_answer": "",
        "agent_history": [],
        "use_rag": False,
        "error": None,
    }