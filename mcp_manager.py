"""
Simple MCP Manager - Direct approach without complex async handling
"""

from mcp_client import VectorStoreMCP, DocumentLoaderMCP
from config import MCP_SERVER_URL, MCP_AUTO_START


class SimpleMCPManager:
    """Simple MCP server management"""
    
    def __init__(self):
        self.vector_client = None
        self.loader_client = None
    
    def get_vector_client(self):
        """Get vector store client"""
        if self.vector_client is None:
            self.vector_client = VectorStoreMCP(
                server_url=MCP_SERVER_URL,
                auto_start_server=MCP_AUTO_START
            )
        return self.vector_client
    
    def get_loader_client(self):
        """Get document loader client"""
        if self.loader_client is None:
            self.loader_client = DocumentLoaderMCP(
                server_url=MCP_SERVER_URL,
                auto_start_server=MCP_AUTO_START
            )
        return self.loader_client
    
    def cleanup(self):
        """Cleanup clients"""
        if self.vector_client:
            self.vector_client._client.close()
        if self.loader_client:
            self.loader_client._client.close()


# Global manager instance
_manager = SimpleMCPManager()

# Convenience functions
def clear_vector_store():
    """Clear vector store"""
    return _manager.get_vector_client().clear_all()

def load_documents(directory_path: str, recursive: bool = True):
    """Load documents"""
    return _manager.get_loader_client().load_directory(directory_path, recursive)

def index_documents(documents):
    """Index documents"""
    return _manager.get_vector_client().add_documents(documents)

def get_vector_stats():
    """Get vector store stats"""
    return _manager.get_vector_client().stats()

def search_documents(query: str, k: int = 3):
    """Search documents"""
    return _manager.get_vector_client().search(query, k)
