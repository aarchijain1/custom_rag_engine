"""
Master Orchestrator Agent
Routes queries to appropriate sub-agents using tool-based routing
"""

from typing import Literal
from agents.shared_state import AgentState
from llm import llm


# Available sub-agents (this list will grow)
AVAILABLE_AGENTS = {
    "faq": "Handles general questions about Toyota Financial Services using RAG",
    "planner": "Creates lease-end timelines and planning (future)",
    "payment": "Calculates payments and financial information (future)",
}


MASTER_PROMPT = """
You are the Master Orchestrator for Toyota Financial Services.

Your job is to analyze the user's question and determine which specialized agent should handle it.

Available Agents:
{agents_list}

User Question: {question}

Analyze the question and respond with ONLY the agent name that should handle it:
- "faq" - For general TFS questions, lease-end information, policies, FAQs
- "planner" - For creating timelines, planning, checklists
- "payment" - For payment calculations, residual values, payoff quotes
- "none" - For greetings or if no agent is needed

Agent:
"""


class MasterAgent:
    """
    Master Orchestrator Agent
    
    Responsibilities:
    - Analyze user queries
    - Route to appropriate sub-agent
    - Provide conversational interface
    - Handle errors gracefully
    """
    
    def __init__(self):
        self.name = "master"
        self.available_agents = AVAILABLE_AGENTS
    
    def route_query(self, state: AgentState) -> AgentState:
        """
        Determine which sub-agent should handle the query
        
        Returns updated state with next_agent set
        """
        # Format agent list for prompt
        agents_list = "\n".join([
            f"- {name}: {desc}" 
            for name, desc in self.available_agents.items()
        ])
        
        # Ask LLM to route
        prompt = MASTER_PROMPT.format(
            agents_list=agents_list,
            question=state["question"]
        )
        
        response = llm.invoke(prompt)
        next_agent = response.content.strip().lower()
        
        # Validate and set next agent
        if next_agent not in self.available_agents and next_agent != "none":
            next_agent = "faq"  # Default to FAQ if invalid
        
        state["next_agent"] = next_agent
        state["agent_history"].append("master")
        
        return state
    
    def should_continue(self, state: AgentState) -> Literal["faq", "planner", "payment", "end"]:
        """
        Routing function for LangGraph
        
        This is called by LangGraph to determine the next node
        """
        next_agent = state.get("next_agent", "none")
        
        if next_agent == "none" or not next_agent:
            return "end"
        
        # Map to actual graph node names
        agent_mapping = {
            "faq": "faq",
            "planner": "planner",
            "payment": "payment",
        }
        
        return agent_mapping.get(next_agent, "end")


# Node function for LangGraph
def master_agent_node(state: AgentState) -> AgentState:
    """
    Master agent node for LangGraph
    """
    master = MasterAgent()
    return master.route_query(state)