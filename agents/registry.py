"""
Agent Registry
Central registry for all sub-agents
Makes it easy to add/remove agents without changing graph code
"""

from typing import Dict, Callable
from agents.shared_state import AgentState
from agents.claude_faq_agent import claude_faq_agent_node
from agents.planner_agent import planner_agent_node
from agents.claude_payment_agent import claude_payment_agent_node


class AgentRegistry:
    """
    Registry of all available sub-agents
    
    Benefits:
    - Single place to register new agents
    - Easy to enable/disable agents
    - No need to modify graph code when adding agents
    """
    
    def __init__(self):
        self._agents: Dict[str, Dict] = {}
        self._register_default_agents()
    
    def _register_default_agents(self):
        """Register the default sub-agents"""
        
        # FAQ Agent (Active - Claude SDK)
        self.register(
            name="faq",
            node_func=claude_faq_agent_node,
            description="Handles general TFS questions using Claude SDK with RAG",
            enabled=True,
            requires_mcp=True,
            mcp_resources=["vector_store"]
        )
        
        # Payment Agent (Active - Claude SDK)
        self.register(
            name="payment",
            node_func=claude_payment_agent_node,
            description="Processes payments, checks payment history, and handles payment-related queries",
            enabled=True,  # ✅ Now enabled
            requires_mcp=False,  # No MCP needed - uses Claude tools directly
            mcp_resources=[]
        )
        
        # Planner Agent (Future)
        self.register(
            name="planner",
            node_func=planner_agent_node,
            description="Creates lease-end timelines and planning",
            enabled=False,  # Set to True when implemented
            requires_mcp=True,
            mcp_resources=["planner"]
        )
    
    def register(
        self,
        name: str,
        node_func: Callable[[AgentState], AgentState],
        description: str,
        enabled: bool = True,
        requires_mcp: bool = False,
        mcp_resources: list = None
    ):
        """
        Register a new agent
        
        Args:
            name: Agent identifier (e.g., "faq", "planner", "payment")
            node_func: LangGraph node function
            description: What this agent does
            enabled: Whether agent is active
            requires_mcp: Whether agent needs MCP resources
            mcp_resources: List of required MCP server names
        """
        self._agents[name] = {
            "name": name,
            "node_func": node_func,
            "description": description,
            "enabled": enabled,
            "requires_mcp": requires_mcp,
            "mcp_resources": mcp_resources or []
        }
    
    def get_agent(self, name: str) -> Dict:
        """Get agent configuration by name"""
        return self._agents.get(name)
    
    def get_enabled_agents(self) -> Dict[str, Dict]:
        """Get all enabled agents"""
        return {
            name: config 
            for name, config in self._agents.items() 
            if config["enabled"]
        }
    
    def get_agent_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all enabled agents"""
        return {
            name: config["description"]
            for name, config in self.get_enabled_agents().items()
        }
    
    def is_enabled(self, name: str) -> bool:
        """Check if an agent is enabled"""
        agent = self._agents.get(name)
        return agent["enabled"] if agent else False
    
    def enable_agent(self, name: str):
        """Enable an agent"""
        if name in self._agents:
            self._agents[name]["enabled"] = True
    
    def disable_agent(self, name: str):
        """Disable an agent"""
        if name in self._agents:
            self._agents[name]["enabled"] = False
    
    def list_agents(self) -> list:
        """List all registered agents"""
        return [
            {
                "name": name,
                "description": config["description"],
                "enabled": config["enabled"],
                "requires_mcp": config["requires_mcp"]
            }
            for name, config in self._agents.items()
        ]


# Global registry instance
agent_registry = AgentRegistry()


# Helper functions for easy access
def get_enabled_agents():
    """Get all enabled agents"""
    return agent_registry.get_enabled_agents()


def get_agent_descriptions():
    """Get descriptions for master agent routing"""
    return agent_registry.get_agent_descriptions()


def is_agent_enabled(name: str) -> bool:
    """Check if an agent is enabled"""
    return agent_registry.is_enabled(name)