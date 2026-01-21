"""
Agent Graph - Builds and manages the LangGraph workflow
This file creates the actual agent graph with all nodes and edges
"""

from langgraph.graph import StateGraph, END
from agent_nodes import (
    AgentState,
    master_agent_node,
    retrieval_node,
    answer_with_rag_node,
    answer_direct_node,
    route_after_master,
    route_after_retrieval
)


def create_rag_agent_graph():
    """
    Create the RAG agent workflow graph
    
    Graph Structure:
    
    START
      ↓
    MASTER AGENT (decides: RAG or DIRECT?)
      ↓         ↓
      ↓       DIRECT ANSWER → END
      ↓
    RETRIEVAL (search documents)
      ↓
    RAG ANSWER (generate with context) → END
    
    Returns:
        Compiled LangGraph
    """
    
    # Initialize the graph with our state schema
    workflow = StateGraph(AgentState)
    
    # ========================================================================
    # Add Nodes
    # ========================================================================
    
    # Master agent - decides the route
    workflow.add_node("master", master_agent_node)
    
    # Retrieval agent - searches documents
    workflow.add_node("retrieve", retrieval_node)
    
    # Answer generators
    workflow.add_node("answer_with_rag", answer_with_rag_node)
    workflow.add_node("answer_direct", answer_direct_node)
    
    # ========================================================================
    # Define Entry Point
    # ========================================================================
    
    # All queries start at the master agent
    workflow.set_entry_point("master")
    
    # ========================================================================
    # Add Conditional Edges (Routing Logic)
    # ========================================================================
    
    # From master agent - route based on decision
    workflow.add_conditional_edges(
        "master",
        route_after_master,
        {
            "retrieve": "retrieve",
            "answer_direct": "answer_direct"
        }
    )
    
    # From retrieval - route based on whether docs were found
    workflow.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "generate_answer": "answer_with_rag",
            "answer_direct": "answer_direct"
        }
    )
    
    # ========================================================================
    # Add Terminal Edges
    # ========================================================================
    
    # Both answer nodes terminate the graph
    workflow.add_edge("answer_with_rag", END)
    workflow.add_edge("answer_direct", END)
    
    # ========================================================================
    # Compile and Return
    # ========================================================================
    
    return workflow.compile()


def visualize_graph(graph, output_file: str = "agent_graph.png"):
    """
    Visualize the agent graph (requires graphviz)
    
    Args:
        graph: Compiled LangGraph
        output_file: Output file path
    """
    try:
        from IPython.display import Image, display
        
        # Get graph visualization
        graph_image = graph.get_graph().draw_mermaid_png()
        
        # Save to file
        with open(output_file, "wb") as f:
            f.write(graph_image)
        
        print(f"✓ Graph visualization saved to: {output_file}")
        
        # Display if in notebook
        try:
            display(Image(graph_image))
        except:
            pass
            
    except ImportError:
        print("⚠ Visualization requires: pip install graphviz")
    except Exception as e:
        print(f"⚠ Could not visualize graph: {str(e)}")


# ============================================================================
# Agent Runner - Convenience wrapper
# ============================================================================

class RAGAgent:
    """
    High-level wrapper for the RAG agent
    Provides a simple interface for querying
    """
    
    def __init__(self):
        """Initialize the agent"""
        print("Initializing RAG Agent...")
        self.graph = create_rag_agent_graph()
        print("✓ RAG Agent ready")
    
    def query(self, user_input: str, verbose: bool = False) -> dict:
        """
        Query the agent
        
        Args:
            user_input: User's question
            verbose: Whether to show detailed processing info
            
        Returns:
            Dictionary with answer and metadata
        """
        from agent_nodes import create_initial_state
        from config import SHOW_RETRIEVED_DOCS
        
        # Create initial state
        initial_state = create_initial_state(user_input)
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        # Prepare response
        response = {
            "answer": result["final_answer"],
            "used_rag": result.get("use_rag", False),
            "error": result.get("error", ""),
            "metadata": {}
        }
        
        # Add retrieval metadata if RAG was used
        if result.get("use_rag") and result.get("retrieved_docs"):
            response["metadata"] = {
                "num_documents": len(result["retrieved_docs"]),
                "sources": result.get("retrieval_metadata", {}).get("sources", [])
            }
            
            # Optionally include retrieved docs
            if SHOW_RETRIEVED_DOCS:
                response["retrieved_documents"] = result["retrieved_docs"]
        
        return response
    
    def chat(self, user_input: str) -> str:
        """
        Simple chat interface - just returns the answer string
        
        Args:
            user_input: User's question
            
        Returns:
            Answer string
        """
        result = self.query(user_input)
        return result["answer"]
    
    def get_stats(self) -> dict:
        """Get agent statistics"""
        from agent_nodes import get_vector_store_stats
        return get_vector_store_stats()


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RAG AGENT GRAPH - Testing")
    print("=" * 70)
    
    # Create agent
    agent = RAGAgent()
    
    # Show stats
    print("\n--- Vector Store Stats ---")
    stats = agent.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test queries
    print("\n--- Test Queries ---")
    
    test_queries = [
        "Hello, how are you?",  # Should use direct answer
        "What is Python?",  # Should use RAG if documents exist
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = agent.query(query, verbose=True)
        print(f"Used RAG: {result['used_rag']}")
        print(f"Answer: {result['answer'][:200]}...")
        if result.get('error'):
            print(f"Error: {result['error']}")
    
    print("\n" + "=" * 70)
    print("Testing complete!")