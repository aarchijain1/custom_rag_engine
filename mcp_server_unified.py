"""
Unified MCP Server - Streaming HTTP Transport (Windows-Compatible)
Uses streaming HTTP with Server-Sent Events for real-time communication
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import json
import sys
import asyncio
from typing import List, Dict, Any, AsyncGenerator

# Import modules
import os
import shutil
from pathlib import Path

# ============================================================================
# CHROMADB CORRUPTION RECOVERY
# ============================================================================

def recover_chromadb_if_needed():
    """Detect and recover from ChromaDB corruption"""
    chroma_path = Path("./chroma_db")
    
    if chroma_path.exists():
        try:
            # Try to initialize vector store to detect corruption
            from vector_store import VectorStore
            vs = VectorStore()
            print("✓ ChromaDB is healthy")
        except Exception as e:
            if "range start index" in str(e) or "slice of length" in str(e):
                print("⚠️ ChromaDB corruption detected, recovering...")
                try:
                    shutil.rmtree(chroma_path)
                    print("✓ Corrupted database removed, will recreate fresh")
                except Exception as cleanup_error:
                    print(f"⚠️ Could not remove corrupted database: {cleanup_error}")
            else:
                print(f"⚠️ Different error detected: {e}")
                raise

# Recover before importing modules
recover_chromadb_if_needed()

from vector_store import VectorStore
from document_loader import DocumentLoader

# ============================================================================
# GLOBAL INSTANCES (Singletons)
# ============================================================================

vector_store = VectorStore()
document_loader = DocumentLoader()

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(title="MCP Unified RAG Tools Server")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_operation(func):
    """Decorator to redirect stdout and handle errors"""
    def wrapper(*args, **kwargs):
        _stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            return func(*args, **kwargs)
        except Exception as e:
            sys.stderr.write(f"Error in {func.__name__}: {str(e)}\n")
            raise
        finally:
            sys.stdout = _stdout
    return wrapper

# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

@safe_operation
def tool_load_file(file_path: str) -> Dict:
    """Load a single file"""
    try:
        return document_loader.load_file(file_path)
    except Exception as e:
        return {"error": str(e), "file_path": file_path}

def tool_load_directory(path: str, recursive: bool = True) -> List[Dict]:
    """Load directory"""
    try:
        result = document_loader.load_directory(path, recursive)
        return result if isinstance(result, list) else []
    except Exception as e:
        sys.stderr.write(f"Error loading directory: {str(e)}\n")
        return []

@safe_operation
def tool_get_supported_extensions() -> List[str]:
    """Get supported extensions"""
    return document_loader.supported_extensions

@safe_operation
def tool_search_documents(query: str, k: int = 3) -> List[Dict]:
    """Search documents"""
    try:
        return vector_store.search(query, k)
    except Exception as e:
        return []

@safe_operation
def tool_add_document(doc_id: str, text: str, metadata: dict) -> dict:
    """Add single document"""
    try:
        vector_store.add_document(doc_id, text, metadata)
        chunks = vector_store._chunk_text(text)
        return {
            "success": True,
            "doc_id": doc_id,
            "chunks_created": len(chunks)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@safe_operation
def tool_add_documents(documents: List[Dict]) -> Dict:
    """Add multiple documents"""
    try:
        return vector_store.add_documents(documents)
    except Exception as e:
        return {"success": False, "error": str(e)}

@safe_operation
def tool_clear_vector_store() -> dict:
    """Clear vector store"""
    try:
        success = vector_store.clear_all()
        return {
            "success": success,
            "message": "Vector store cleared" if success else "Failed to clear"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@safe_operation
def tool_get_vector_store_stats() -> dict:
    """Get stats"""
    try:
        return vector_store.get_stats()
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# STREAMING ENDPOINTS
# ============================================================================

async def generate_sse_response(data: dict) -> AsyncGenerator[str, None]:
    """Generate Server-Sent Events response"""
    yield f"data: {json.dumps(data)}\n\n"

@app.post("/tools/call")
async def call_tool_stream(request: Request):
    """
    Handle MCP tool calls with streaming response
    
    Request format:
    {
        "name": "tool_name",
        "arguments": {...}
    }
    """
    try:
        data = await request.json()
        tool_name = data.get("name")
        arguments = data.get("arguments", {})
        
        if not tool_name:
            error_data = {"error": "Missing tool name", "status": "error"}
            return StreamingResponse(
                generate_sse_response(error_data),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        
        async def generate_tool_response():
            try:
                # Send start event
                yield f"data: {json.dumps({'status': 'started', 'tool': tool_name})}\n\n"
                
                # Route to appropriate tool and execute
                if tool_name == "load_file":
                    result = tool_load_file(arguments["file_path"])
                
                elif tool_name == "load_directory":
                    result = tool_load_directory(
                        arguments["path"],
                        arguments.get("recursive", True)
                    )
                
                elif tool_name == "get_supported_extensions":
                    result = tool_get_supported_extensions()
                
                elif tool_name == "search_documents":
                    result = tool_search_documents(
                        arguments["query"],
                        arguments.get("k", 3)
                    )
                
                elif tool_name == "add_document":
                    result = tool_add_document(
                        arguments["doc_id"],
                        arguments["text"],
                        arguments.get("metadata", {})
                    )
                
                elif tool_name == "add_documents":
                    result = tool_add_documents(arguments["documents"])
                
                elif tool_name == "clear_vector_store":
                    result = tool_clear_vector_store()
                
                elif tool_name == "get_vector_store_stats":
                    result = tool_get_vector_store_stats()
                
                else:
                    error_data = {"error": f"Unknown tool: {tool_name}", "status": "error"}
                    yield f"data: {json.dumps(error_data)}\n\n"
                    return
                
                # Send result event
                response_data = {"result": result, "status": "completed", "tool": tool_name}
                yield f"data: {json.dumps(response_data)}\n\n"
                
            except Exception as e:
                error_data = {"error": str(e), "status": "error", "tool": tool_name}
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(
            generate_tool_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache", 
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )
    
    except Exception as e:
        error_data = {"error": str(e), "status": "error"}
        return StreamingResponse(
            generate_sse_response(error_data),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )

@app.post("/tools/call/simple")
async def call_tool_simple(request: Request):
    """
    Simple non-streaming endpoint for backward compatibility
    """
    try:
        data = await request.json()
        tool_name = data.get("name")
        arguments = data.get("arguments", {})
        
        if not tool_name:
            raise HTTPException(status_code=400, detail="Missing tool name")
        
        # Route to appropriate tool
        if tool_name == "load_file":
            result = tool_load_file(arguments["file_path"])
        
        elif tool_name == "load_directory":
            result = tool_load_directory(
                arguments["path"],
                arguments.get("recursive", True)
            )
        
        elif tool_name == "get_supported_extensions":
            result = tool_get_supported_extensions()
        
        elif tool_name == "search_documents":
            result = tool_search_documents(
                arguments["query"],
                arguments.get("k", 3)
            )
        
        elif tool_name == "add_document":
            result = tool_add_document(
                arguments["doc_id"],
                arguments["text"],
                arguments.get("metadata", {})
            )
        
        elif tool_name == "add_documents":
            result = tool_add_documents(arguments["documents"])
        
        elif tool_name == "clear_vector_store":
            result = tool_clear_vector_store()
        
        elif tool_name == "get_vector_store_stats":
            result = tool_get_vector_store_stats()
        
        else:
            raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_name}")
        
        return JSONResponse({"result": result})
    
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )

# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "ok",
        "server": "MCP Unified RAG Tools",
        "stats": tool_get_vector_store_stats()
    })

# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" UNIFIED MCP SERVER - STREAMING HTTP TRANSPORT")
    print("=" * 60)
    print("Server: http://127.0.0.1:8765")
    print("Health Check: http://127.0.0.1:8765/")
    print("Streaming Tool Endpoint: http://127.0.0.1:8765/tools/call")
    print("Simple Tool Endpoint: http://127.0.0.1:8765/tools/call/simple")
    print("\n Available Tools:")
    print("  Document Loading (load_file, load_directory)")
    print("  Vector Store (search, add, clear)")
    print("  Automatic Chunking & Embedding")
    print("  Real-time Streaming Responses")
    print("\n" + "=" * 60)
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8765,
        log_level="error"
    )