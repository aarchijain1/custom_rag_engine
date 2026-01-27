"""
MCP Server for Vector Store Operations
Exposes ChromaDB via Model Context Protocol (stdio)
FIXED: Redirects VectorStore stdout to stderr
"""

from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Any
import json
import sys
import io

# Redirect stdout to stderr during imports to suppress VectorStore prints
_original_stdout = sys.stdout
sys.stdout = sys.stderr

try:
    from vector_store import VectorStore
finally:
    sys.stdout = _original_stdout

# Initialize FastMCP server
mcp = FastMCP("vector-store")

# Initialize vector store (singleton) with stdout redirected
sys.stdout = sys.stderr
vector_store = VectorStore()
sys.stdout = _original_stdout


# ============================================================================
# MCP TOOLS
# ============================================================================

@mcp.tool()
def search_documents(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Search for relevant documents in the vector store.
    
    Args:
        query: Search query string
        k: Number of results to return (default: 3)
        
    Returns:
        List of relevant document chunks with metadata
    """
    try:
        # Redirect stdout during operation
        _stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            results = vector_store.search(query, k)
        finally:
            sys.stdout = _stdout
        return results
    except Exception as e:
        return {"error": str(e), "results": []}


@mcp.tool()
def add_document(doc_id: str, text: str, metadata: dict = None) -> dict:
    """
    Add a single document to the vector store.
    
    Args:
        doc_id: Unique document identifier
        text: Document text content
        metadata: Optional metadata dictionary
        
    Returns:
        Success status and chunk count
    """
    try:
        if metadata is None:
            metadata = {}
        
        # Redirect stdout during operation
        _stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            vector_store.add_document(doc_id, text, metadata)
            chunks = vector_store._chunk_text(text)
        finally:
            sys.stdout = _stdout
        
        return {
            "success": True,
            "doc_id": doc_id,
            "chunks_created": len(chunks)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
def add_documents(documents: List[Dict]) -> Dict:
    """
    Add multiple documents to the vector store.
    
    Args:
        documents: List of document dictionaries with 'id', 'text', and optional 'metadata'
        
    Returns:
        Summary of indexing operation
    """
    try:
        # Redirect stdout during operation
        _stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            result = vector_store.add_documents(documents)
        finally:
            sys.stdout = _stdout
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
def clear_vector_store() -> dict:
    """
    Clear all documents from the vector store.
    
    Returns:
        Success status
    """
    try:
        # Redirect stdout during operation
        _stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            success = vector_store.clear_all()
        finally:
            sys.stdout = _stdout
        return {
            "success": success,
            "message": "Vector store cleared" if success else "Failed to clear"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
def get_vector_store_stats() -> dict:
    """
    Get statistics about the vector store.
    
    Returns:
        Dictionary with total chunks, collection name, and embedding model
    """
    try:
        # Redirect stdout during operation
        _stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            stats = vector_store.get_stats()
        finally:
            sys.stdout = _stdout
        return stats
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# RESOURCES (Optional - for exposing store info)
# ============================================================================

@mcp.resource("vector://stats")
def get_stats_resource() -> str:
    """Expose vector store statistics as a resource"""
    # Redirect stdout during operation
    _stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        stats = vector_store.get_stats()
    finally:
        sys.stdout = _stdout
    return json.dumps(stats, indent=2)


# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == "__main__":
    # Ensure all output goes to stderr except JSON-RPC
    import warnings
    warnings.filterwarnings("ignore")
    
    # Run the MCP server over stdio
    mcp.run(transport="stdio")