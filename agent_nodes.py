"""
Agent Nodes - Individual functions that make up the agent workflow
Each node performs a specific task in the RAG pipeline
"""

from typing import TypedDict, Annotated, Sequence
import operator
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    TEMPERATURE,
    MAX_OUTPUT_TOKENS,
    CLASSIFICATION_PROMPT,
    RAG_ANSWER_PROMPT,
    DIRECT_ANSWER_PROMPT,
    N_RETRIEVAL_RESULTS,
    VERBOSE_MODE
)
from vector_store import VectorStore


# ============================================================================
# Agent State Definition
# ============================================================================

class AgentState(TypedDict):
    """
    State that flows through the agent graph
    This gets passed between nodes and accumulates information
    """
    # Conversation messages
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    
    # Current user query
    user_query: str
    
    # Retrieved documents from vector store
    retrieved_docs: list
    
    # Metadata about retrieved docs
    retrieval_metadata: dict
    
    # Next action to take (for routing)
    next_action: str
    
    # Final answer to user
    final_answer: str
    
    # Additional context or flags
    use_rag: bool
    error: str


# ============================================================================
# Initialize Components
# ============================================================================

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=TEMPERATURE,
    max_output_tokens=MAX_OUTPUT_TOKENS
)

# Initialize vector store
vector_store = VectorStore()


# ============================================================================
# Node Functions
# ============================================================================

def master_agent_node(state: AgentState) -> AgentState:
    """
    Master Agent Node
    Analyzes the user query and decides whether to use RAG or answer directly
    
    This is the routing logic - it determines the path through the graph
    """
    user_query = state["user_query"]
    
    if VERBOSE_MODE:
        print(f"\n[Master Agent] Analyzing query: {user_query[:50]}...")
    
    try:
        # Use LLM to classify the query
        classification_prompt = CLASSIFICATION_PROMPT.format(query=user_query)
        
        response = llm.invoke([HumanMessage(content=classification_prompt)])
        decision = response.content.strip().upper()
        
        # Determine next action based on classification
        if "RAG" in decision:
            state["next_action"] = "retrieve"
            state["use_rag"] = True
            if VERBOSE_MODE:
                print("[Master Agent] Decision: Use RAG")
        else:
            state["next_action"] = "answer_direct"
            state["use_rag"] = False
            if VERBOSE_MODE:
                print("[Master Agent] Decision: Answer directly")
        
        state["error"] = ""
        
    except Exception as e:
        print(f"[Master Agent] Error: {str(e)}")
        state["error"] = f"Classification error: {str(e)}"
        state["next_action"] = "answer_direct"  # Fallback to direct answer
        state["use_rag"] = False
    
    return state


def retrieval_node(state: AgentState) -> AgentState:
    """
    Retrieval Node
    Searches the vector database for relevant documents
    
    This is the RAG sub-agent that specializes in document retrieval
    """
    user_query = state["user_query"]
    
    if VERBOSE_MODE:
        print(f"\n[Retrieval Agent] Searching for: {user_query[:50]}...")
    
    try:
        # Query vector store
        results = vector_store.query(user_query, n_results=N_RETRIEVAL_RESULTS)
        
        # Store retrieved documents
        state["retrieved_docs"] = results["documents"]
        state["retrieval_metadata"] = {
            "num_results": len(results["documents"]),
            "distances": results.get("distances", []),
            "sources": [m.get("source", "unknown") for m in results.get("metadatas", [])]
        }
        
        if VERBOSE_MODE:
            print(f"[Retrieval Agent] Found {len(results['documents'])} relevant chunks")
        
        # Set next action
        if results["documents"]:
            state["next_action"] = "generate_answer"
        else:
            # No documents found - fall back to direct answer
            if VERBOSE_MODE:
                print("[Retrieval Agent] No relevant documents found, falling back to direct answer")
            state["next_action"] = "answer_direct"
        
        state["error"] = ""
        
    except Exception as e:
        print(f"[Retrieval Agent] Error: {str(e)}")
        state["error"] = f"Retrieval error: {str(e)}"
        state["next_action"] = "answer_direct"  # Fallback
        state["retrieved_docs"] = []
    
    return state


def answer_with_rag_node(state: AgentState) -> AgentState:
    """
    RAG Answer Generation Node
    Generates an answer using the retrieved documents as context
    """
    user_query = state["user_query"]
    retrieved_docs = state["retrieved_docs"]
    
    if VERBOSE_MODE:
        print(f"\n[RAG Generator] Generating answer with {len(retrieved_docs)} documents...")
    
    try:
        # Combine retrieved documents into context
        context = "\n\n---\n\n".join([
            f"Document {i+1}:\n{doc}" 
            for i, doc in enumerate(retrieved_docs)
        ])
        
        # Create prompt with context
        prompt = RAG_ANSWER_PROMPT.format(
            context=context,
            query=user_query
        )
        
        # Generate answer
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content
        
        # Store results
        state["final_answer"] = answer
        state["messages"].append(AIMessage(content=answer))
        state["next_action"] = "end"
        state["error"] = ""
        
        if VERBOSE_MODE:
            print(f"[RAG Generator] Answer generated ({len(answer)} chars)")
        
    except Exception as e:
        print(f"[RAG Generator] Error: {str(e)}")
        state["error"] = f"Answer generation error: {str(e)}"
        state["final_answer"] = f"I encountered an error generating the answer: {str(e)}"
        state["next_action"] = "end"
    
    return state


def answer_direct_node(state: AgentState) -> AgentState:
    """
    Direct Answer Node
    Answers the query directly without using retrieved documents
    Used for general knowledge questions, greetings, etc.
    """
    user_query = state["user_query"]
    
    if VERBOSE_MODE:
        print(f"\n[Direct Answer] Generating direct answer...")
    
    try:
        # Create prompt
        prompt = DIRECT_ANSWER_PROMPT.format(query=user_query)
        
        # Generate answer
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content
        
        # Store results
        state["final_answer"] = answer
        state["messages"].append(AIMessage(content=answer))
        state["next_action"] = "end"
        state["error"] = ""
        
        if VERBOSE_MODE:
            print(f"[Direct Answer] Answer generated ({len(answer)} chars)")
        
    except Exception as e:
        print(f"[Direct Answer] Error: {str(e)}")
        state["error"] = f"Answer generation error: {str(e)}"
        state["final_answer"] = f"I encountered an error: {str(e)}"
        state["next_action"] = "end"
    
    return state


# ============================================================================
# Routing Functions
# ============================================================================

def route_after_master(state: AgentState) -> str:
    """
    Routing function after master agent
    Decides which node to go to based on state
    """
    return state["next_action"]


def route_after_retrieval(state: AgentState) -> str:
    """
    Routing function after retrieval
    Goes to RAG answer if docs found, otherwise direct answer
    """
    return state["next_action"]


# ============================================================================
# Utility Functions
# ============================================================================

def create_initial_state(user_input: str) -> AgentState:
    """
    Create the initial state for a new query
    
    Args:
        user_input: User's question/query
        
    Returns:
        Initial AgentState
    """
    return {
        "messages": [HumanMessage(content=user_input)],
        "user_query": user_input,
        "retrieved_docs": [],
        "retrieval_metadata": {},
        "next_action": "",
        "final_answer": "",
        "use_rag": False,
        "error": ""
    }


def get_vector_store_stats() -> dict:
    """Get statistics from the vector store"""
    return vector_store.get_stats()