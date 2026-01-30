"""
Planner Sub-Agent (Placeholder)
Creates lease-end timelines and planning
"""

from agents.shared_state import AgentState
from llm import llm


class PlannerAgent:
    """
    Planner Sub-Agent (To Be Implemented)
    
    Future Responsibilities:
    - Create lease-end timelines
    - Generate action checklists
    - Provide milestone tracking
    """
    
    def __init__(self):
        self.name = "planner"
    
    def process(self, state: AgentState) -> AgentState:
        """
        Placeholder implementation
        """
        state["final_answer"] = (
            "The planner agent is under development. "
            "For now, please ask general questions about lease-end options."
        )
        state["agent_history"].append("planner")
        return state


# Node function for LangGraph
def planner_agent_node(state: AgentState) -> AgentState:
    """
    Planner agent node for LangGraph
    """
    planner_agent = PlannerAgent()
    return planner_agent.process(state)