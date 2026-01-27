"""
MCP Client for RAG Agent
Communicates with MCP servers via stdio
FIXED: Proper handling of list responses from MCP tools
"""

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import List, Dict, Any
import asyncio
import json


class MCPVectorStoreClient:
    """Client for vector store MCP server"""
    
    def __init__(self, server_script_path: str = "mcp_server_vector.py"):
        self.server_script = server_script_path
        self.session = None
        self.context_manager = None
        
    async def connect(self):
        """Connect to the MCP server"""
        server_params = StdioServerParameters(
            command="python",
            args=[self.server_script],
        )
        
        # stdio_client is a context manager, not a regular async function
        self.context_manager = stdio_client(server_params)
        read, write = await self.context_manager.__aenter__()
        
        self.session = ClientSession(read, write)
        await self.session.__aenter__()
        await self.session.initialize()
        
    async def disconnect(self):
        """Disconnect from the MCP server"""
        if self.session:
            await self.session.__aexit__(None, None, None)
        if self.context_manager:
            await self.context_manager.__aexit__(None, None, None)
    
    def _parse_tool_result(self, result) -> Any:
        """Parse MCP tool call result - handles both single values and lists"""
        if not result or not result.content:
            return None
        
        # Collect all text content items
        all_text = []
        for content_item in result.content:
            # Check if it's a text content type
            if hasattr(content_item, 'type') and content_item.type == 'text':
                if hasattr(content_item, 'text'):
                    all_text.append(content_item.text)
        
        # If no text found, return None
        if not all_text:
            return None
        
        # If single item, parse it
        if len(all_text) == 1:
            text = all_text[0]
            # Try to parse as JSON
            if text and (text.startswith('{') or text.startswith('[')):
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text
            return text
        
        # Multiple items - try to parse each and return as list
        parsed_items = []
        for text in all_text:
            if text and (text.startswith('{') or text.startswith('[')):
                try:
                    parsed_items.append(json.loads(text))
                except json.JSONDecodeError:
                    parsed_items.append(text)
            else:
                parsed_items.append(text)
        
        return parsed_items if len(parsed_items) > 1 else parsed_items[0]
    
    async def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for documents"""
        result = await self.session.call_tool("search_documents", {
            "query": query,
            "k": k
        })
        parsed = self._parse_tool_result(result)
        return parsed if parsed is not None else []
    
    async def add_document(self, doc_id: str, text: str, metadata: Dict) -> Dict:
        """Add a single document"""
        result = await self.session.call_tool("add_document", {
            "doc_id": doc_id,
            "text": text,
            "metadata": metadata
        })
        parsed = self._parse_tool_result(result)
        return parsed if parsed is not None else {}
    
    async def add_documents(self, documents: List[Dict]) -> Dict:
        """Add multiple documents"""
        result = await self.session.call_tool("add_documents", {
            "documents": documents
        })
        parsed = self._parse_tool_result(result)
        return parsed if parsed is not None else {}
    
    async def clear_all(self) -> bool:
        """Clear all documents"""
        result = await self.session.call_tool("clear_vector_store", {})
        parsed = self._parse_tool_result(result)
        if isinstance(parsed, dict):
            return parsed.get("success", False)
        return False
    
    async def stats(self) -> Dict:
        """Get vector store statistics"""
        result = await self.session.call_tool("get_vector_store_stats", {})
        parsed = self._parse_tool_result(result)
        return parsed if parsed is not None else {}


class MCPDocumentLoaderClient:
    """Client for document loader MCP server"""
    
    def __init__(self, server_script_path: str = "mcp_server_documents.py"):
        self.server_script = server_script_path
        self.session = None
        self.context_manager = None
        
    async def connect(self):
        """Connect to the MCP server"""
        server_params = StdioServerParameters(
            command="python",
            args=[self.server_script],
        )
        
        # stdio_client is a context manager
        self.context_manager = stdio_client(server_params)
        read, write = await self.context_manager.__aenter__()
        
        self.session = ClientSession(read, write)
        await self.session.__aenter__()
        await self.session.initialize()
        
    async def disconnect(self):
        """Disconnect from the MCP server"""
        if self.session:
            await self.session.__aexit__(None, None, None)
        if self.context_manager:
            await self.context_manager.__aexit__(None, None, None)
    
    def _parse_tool_result(self, result) -> Any:
        """Parse MCP tool call result - handles both single values and lists"""
        if not result or not result.content:
            return None
        
        # Collect all text content items
        all_text = []
        for content_item in result.content:
            if hasattr(content_item, 'type') and content_item.type == 'text':
                if hasattr(content_item, 'text'):
                    all_text.append(content_item.text)
        
        # If no text found, return None
        if not all_text:
            return None
        
        # If single item, parse it
        if len(all_text) == 1:
            text = all_text[0]
            # Try to parse as JSON
            if text and (text.startswith('{') or text.startswith('[')):
                try:
                    parsed = json.loads(text)
                    # IMPORTANT: If we parsed a list, return the list, not wrapped
                    return parsed
                except json.JSONDecodeError:
                    return text
            return text
        
        # Multiple items - try to parse each
        parsed_items = []
        for text in all_text:
            if text and (text.startswith('{') or text.startswith('[')):
                try:
                    parsed_items.append(json.loads(text))
                except json.JSONDecodeError:
                    parsed_items.append(text)
            else:
                parsed_items.append(text)
        
        return parsed_items if len(parsed_items) > 1 else parsed_items[0]
    
    async def load_directory(self, path: str, recursive: bool = True) -> List[Dict]:
        """Load documents from directory"""
        result = await self.session.call_tool("load_directory", {
            "path": path,
            "recursive": recursive
        })
        parsed = self._parse_tool_result(result)
        
        # Ensure we return a list
        if parsed is None:
            return []
        if isinstance(parsed, list):
            return parsed
        # If it's a single dict, wrap it
        if isinstance(parsed, dict):
            return [parsed]
        return []
    
    async def load_file(self, file_path: str) -> Dict:
        """Load a single file"""
        result = await self.session.call_tool("load_file", {
            "file_path": file_path
        })
        parsed = self._parse_tool_result(result)
        return parsed if parsed is not None else {}


# ============================================================================
# Synchronous Wrappers (for compatibility with existing code)
# ============================================================================

class VectorStoreMCP:
    """Synchronous wrapper for MCP vector store client"""
    
    def __init__(self):
        self._client = None
        self._loop = None
        
    def _get_or_create_loop(self):
        """Get or create event loop"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    
    def _ensure_connection(self):
        """Ensure connection is established"""
        if self._client is None:
            self._client = MCPVectorStoreClient()
            loop = self._get_or_create_loop()
            loop.run_until_complete(self._client.connect())
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        self._ensure_connection()
        loop = self._get_or_create_loop()
        return loop.run_until_complete(self._client.search(query, k))
    
    def add_documents(self, documents: List[Dict]) -> Dict:
        self._ensure_connection()
        loop = self._get_or_create_loop()
        return loop.run_until_complete(self._client.add_documents(documents))
    
    def add_document(self, doc_id: str, text: str, metadata: Dict):
        self._ensure_connection()
        loop = self._get_or_create_loop()
        return loop.run_until_complete(self._client.add_document(doc_id, text, metadata))
    
    def clear_all(self) -> bool:
        self._ensure_connection()
        loop = self._get_or_create_loop()
        return loop.run_until_complete(self._client.clear_all())
    
    def stats(self) -> Dict:
        self._ensure_connection()
        loop = self._get_or_create_loop()
        return loop.run_until_complete(self._client.stats())
    
    def __del__(self):
        """Cleanup on deletion"""
        if self._client:
            try:
                loop = self._get_or_create_loop()
                loop.run_until_complete(self._client.disconnect())
            except:
                pass


class DocumentLoaderMCP:
    """Synchronous wrapper for MCP document loader client"""
    
    def __init__(self):
        self._client = None
        self._loop = None
        
    def _get_or_create_loop(self):
        """Get or create event loop"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    
    def _ensure_connection(self):
        """Ensure connection is established"""
        if self._client is None:
            self._client = MCPDocumentLoaderClient()
            loop = self._get_or_create_loop()
            loop.run_until_complete(self._client.connect())
    
    def load_directory(self, path: str, recursive: bool = True) -> List[Dict]:
        self._ensure_connection()
        loop = self._get_or_create_loop()
        return loop.run_until_complete(self._client.load_directory(path, recursive))
    
    def load_file(self, file_path: str) -> Dict:
        self._ensure_connection()
        loop = self._get_or_create_loop()
        return loop.run_until_complete(self._client.load_file(file_path))
    
    def __del__(self):
        """Cleanup on deletion"""
        if self._client:
            try:
                loop = self._get_or_create_loop()
                loop.run_until_complete(self._client.disconnect())
            except:
                pass