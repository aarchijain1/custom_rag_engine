"""
Payment Sub-Agent (Placeholder)
Calculates payments and financial information
"""

from agents.shared_state import AgentState
from llm import llm


class PaymentAgent:
    """
    Payment Sub-Agent (To Be Implemented)
    
    Future Responsibilities:
    - Calculate payment amounts
    - Compute residual values
    - Generate payoff quotes
    - Explain payment options
    """
    
    def __init__(self):
        self.name = "payment"
    
    def process(self, state: AgentState) -> AgentState:
        """
        Placeholder implementation
        """
        state["final_answer"] = (
            "The payment calculator agent is under development. "
            "For general payment information, please ask the FAQ agent."
        )
        state["agent_history"].append("payment")
        return state


# Node function for LangGraph
def payment_agent_node(state: AgentState) -> AgentState:
    """
    Payment agent node for LangGraph
    """
    payment_agent = PaymentAgent()
    return payment_agent.process(state)