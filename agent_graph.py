"""
Multi-Agent LangGraph Workflow
Scalable architecture using Agent Registry
"""

from langgraph.graph import StateGraph, END
from agents.shared_state import AgentState, create_initial_state
from agents.master_agent import master_agent_node, MasterAgent
from agents.registry import agent_registry, get_enabled_agents


def build_multi_agent_graph():
    """
    Build multi-agent graph dynamically from registry
    
    This approach scales automatically as you add agents:
    1. Register agent in registry.py
    2. No need to modify this file!
    """
    
    workflow = StateGraph(AgentState)
    
    # Add master agent (always present)
    workflow.add_node("master", master_agent_node)
    workflow.set_entry_point("master")
    
    # Dynamically add all enabled sub-agents from registry
    enabled_agents = get_enabled_agents()
    
    for agent_name, agent_config in enabled_agents.items():
        workflow.add_node(agent_name, agent_config["node_func"])
        # All agents end after completing their work
        workflow.add_edge(agent_name, END)
    
    # Master agent routing
    # Uses the routing function from MasterAgent
    master = MasterAgent()
    
    # Build routing dictionary dynamically
    routing_dict = {
        agent_name: agent_name 
        for agent_name in enabled_agents.keys()
    }
    routing_dict["end"] = END
    
    workflow.add_conditional_edges(
        "master",
        master.should_continue,
        routing_dict
    )
    
    return workflow.compile()


class MultiAgentRAGSystem:
    """
    Multi-Agent RAG System with Dynamic Agent Loading
    
    Features:
    - Master agent routes to specialized sub-agents
    - Sub-agents registered in agent registry
    - Easy to add new agents without modifying graph
    - MCP-based resource access
    """
    
    def __init__(self):
        print("=" * 60)
        print("🤖 INITIALIZING MULTI-AGENT RAG SYSTEM")
        print("=" * 60)
        
        # Build graph from registry
        self.graph = build_multi_agent_graph()
        
        # Show registered agents
        print("\n📋 Registered Agents:")
        print("  ✅ Master Agent (orchestrator)")
        
        for agent_info in agent_registry.list_agents():
            status = "✅" if agent_info["enabled"] else "🚧"
            print(f"  {status} {agent_info['name'].upper()} Agent - {agent_info['description']}")
        
        print("\n" + "=" * 60)
        print("✓ Multi-Agent System Ready")
        print("=" * 60)
    
    def query(self, user_input: str, verbose: bool = False) -> dict:
        """
        Query the multi-agent system
        
        Args:
            user_input: User's question
            verbose: Show agent routing details
            
        Returns:
            dict with answer, agent_history, used_rag, error
        """
        # Create initial state
        state = create_initial_state(user_input)
        
        # Run through graph
        result = self.graph.invoke(state)
        
        # Prepare response
        response = {
            "answer": result["final_answer"],
            "agent_history": result["agent_history"],
            "used_rag": result["use_rag"],
            "error": result.get("error"),
        }
        
        # Verbose output
        if verbose:
            print(f"\n[DEBUG] Agent Path: {' → '.join(result['agent_history'])}")
            print(f"[DEBUG] Used RAG: {result['use_rag']}")
            if result.get("error"):
                print(f"[DEBUG] Error: {result['error']}")
        
        return response
    
    def chat(self, user_input: str) -> str:
        """
        Simple chat interface - returns just the answer
        """
        result = self.query(user_input)
        return result["answer"]
    
    def get_stats(self):
        """Get system statistics"""
        from mcp_client import VectorStoreMCP
        
        vector_mcp = VectorStoreMCP()
        vector_stats = vector_mcp.stats()
        
        agent_stats = {
            "total_agents": len(agent_registry.list_agents()),
            "enabled_agents": len(agent_registry.get_enabled_agents()),
            "agents": agent_registry.list_agents()
        }
        
        return {
            "vector_store": vector_stats,
            "agents": agent_stats
        }
    
    def list_agents(self):
        """List all registered agents"""
        return agent_registry.list_agents()


# Alias for backward compatibility
RAGAgent = MultiAgentRAGSystem


# Example usage
if __name__ == "__main__":
    # Initialize system
    agent_system = MultiAgentRAGSystem()
    
    # Test queries
    test_queries = [
        "What are my lease-end options?",  # Should route to FAQ
        "Create a timeline",                # Should route to Planner
        "Calculate my payoff",              # Should route to Payment
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        result = agent_system.query(query, verbose=True)
        print(f"\nAnswer:\n{result['answer']}")