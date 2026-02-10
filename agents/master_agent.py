"""
Master Orchestrator Agent
Routes queries to appropriate sub-agents using tool-based routing
"""

from typing import Literal
from agents.shared_state import AgentState
from agents.registry import get_enabled_agents
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

GREETING_PROMPT = """You are a friendly Toyota Financial Services assistant. The user said: "{question}"

Respond with a brief, welcoming message (1-2 sentences). Keep it natural and helpful."""


def _parse_agent_from_response(text: str) -> str:
    """Extract agent name from LLM response (handles 'faq', 'faq.', 'The agent is faq', etc.)"""
    text = text.strip().lower()
    valid = {"faq", "planner", "payment", "none"}
    words = text.replace(".", " ").replace(",", " ").split()
    for w in words:
        if w in valid:
            return w
    return "faq"  # default


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
        next_agent = _parse_agent_from_response(response.content)
        
        # Validate and set next agent
        if next_agent not in self.available_agents and next_agent != "none":
            next_agent = "faq"  # Default to FAQ if invalid
        
        # Only route to enabled agents; fallback to faq for disabled ones (e.g. planner)
        enabled = set(get_enabled_agents().keys())
        if next_agent != "none" and next_agent not in enabled:
            next_agent = "faq"
        
        state["next_agent"] = next_agent
        state["agent_history"].append("master")
        
        # When routing to "none" (greetings/casual), provide a response so user doesn't get empty output
        if next_agent == "none":
            greeting_prompt = GREETING_PROMPT.format(question=state["question"])
            greeting_response = llm.invoke(greeting_prompt)
            state["final_answer"] = greeting_response.content.strip()
        
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