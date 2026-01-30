"""
HTTP MCP Client - Windows-Compatible
Simple HTTP client for the MCP server
"""

import httpx
import json
from typing import List, Dict, Any, Optional
import asyncio
import subprocess
import time
import sys
import platform


class UnifiedMCPClient:
    """HTTP-based client for the unified MCP server"""
    
    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8765",
        auto_start_server: bool = False,
        server_script: str = "mcp_server_unified.py"
    ):
        self.server_url = server_url.rstrip('/')
        self.health_url = f"{self.server_url}/"
        self.tools_url = f"{self.server_url}/tools/call/simple"
        self.auto_start = auto_start_server
        self.server_script = server_script
        self.server_process = None
        self.is_windows = platform.system() == "Windows"
    
    async def _check_health(self) -> bool:
        """Check if server is accessible"""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(self.health_url)
                return response.status_code == 200
        except:
            return False
    
    async def _ensure_server_running(self):
        """Ensure the server is running"""
        if not self.auto_start:
            return
        
        # Check if already running
        if await self._check_health():
            return
        
        # Start server
        print("🚀 Auto-starting MCP server...")
        
        if self.is_windows:
            self.server_process = subprocess.Popen(
                [sys.executable, self.server_script],
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            self.server_process = subprocess.Popen(
                [sys.executable, self.server_script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
        
        # Wait for server
        print("  Waiting", end="", flush=True)
        for i in range(30):
            await asyncio.sleep(1)
            print(".", end="", flush=True)
            
            if await self._check_health():
                print("\n✓ Server started successfully")
                return
            
            # Check if process died
            if self.server_process.poll() is not None:
                print("\n✗ Server process terminated")
                raise RuntimeError("Server failed to start")
        
        print("\n✗ Server startup timeout")
        raise RuntimeError("Failed to start server after 30 seconds")
    
    async def _call_tool(self, tool_name: str, arguments: dict = None) -> Any:
        """Call a tool on the server"""
        await self._ensure_server_running()
        
        arguments = arguments or {}
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                self.tools_url,
                json={"name": tool_name, "arguments": arguments}
            )
            
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
            
            data = response.json()
            
            if "error" in data:
                raise Exception(data["error"])
            
            return data.get("result")
    
    # ========== DOCUMENT LOADING ==========
    
    async def load_file(self, file_path: str) -> Dict:
        return await self._call_tool("load_file", {"file_path": file_path})
    
    async def load_directory(self, path: str, recursive: bool = True) -> List[Dict]:
        result = await self._call_tool("load_directory", {
            "path": path,
            "recursive": recursive
        })
        return result if isinstance(result, list) else []
    
    async def get_supported_extensions(self) -> List[str]:
        return await self._call_tool("get_supported_extensions")
    
    # ========== VECTOR STORE ==========
    
    async def search(self, query: str, k: int = 3) -> List[Dict]:
        return await self._call_tool("search_documents", {"query": query, "k": k})
    
    async def add_document(self, doc_id: str, text: str, metadata: Dict = None) -> Dict:
        return await self._call_tool("add_document", {
            "doc_id": doc_id,
            "text": text,
            "metadata": metadata or {}
        })
    
    async def add_documents(self, documents: List[Dict]) -> Dict:
        return await self._call_tool("add_documents", {"documents": documents})
    
    async def clear_all(self) -> bool:
        result = await self._call_tool("clear_vector_store")
        return result.get("success", False) if isinstance(result, dict) else False
    
    async def stats(self) -> Dict:
        return await self._call_tool("get_vector_store_stats")
    
    async def close(self):
        """Cleanup"""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except:
                pass


# ============================================================================
# SYNCHRONOUS WRAPPERS
# ============================================================================

class VectorStoreMCP:
    """Synchronous wrapper for vector store operations"""
    
    def __init__(self, server_url="http://127.0.0.1:8765", auto_start_server=False):
        self._client = UnifiedMCPClient(server_url, auto_start_server)
    
    def _run(self, coro):
        """Run async code synchronously"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        return self._run(self._client.search(query, k))
    
    def add_document(self, doc_id: str, text: str, metadata: Dict):
        return self._run(self._client.add_document(doc_id, text, metadata))
    
    def add_documents(self, documents: List[Dict]) -> Dict:
        return self._run(self._client.add_documents(documents))
    
    def clear_all(self) -> bool:
        return self._run(self._client.clear_all())
    
    def stats(self) -> Dict:
        return self._run(self._client.stats())


class DocumentLoaderMCP:
    """Synchronous wrapper for document loader"""
    
    def __init__(self, server_url="http://127.0.0.1:8765", auto_start_server=False):
        self._client = UnifiedMCPClient(server_url, auto_start_server)
    
    def _run(self, coro):
        """Run async code synchronously"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    
    def load_file(self, file_path: str) -> Dict:
        return self._run(self._client.load_file(file_path))
    
    def load_directory(self, path: str, recursive: bool = True) -> List[Dict]:
        return self._run(self._client.load_directory(path, recursive))


if __name__ == "__main__":
    # Test
    print("Testing MCP client...")
    client = VectorStoreMCP(auto_start_server=True)
    
    try:
        stats = client.stats()
        print(f"✓ Stats: {stats}")
    except Exception as e:
        print(f"✗ Error: {e}")