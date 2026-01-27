"""
MCP Server for Document Loader Operations
Exposes filesystem document loading via Model Context Protocol (stdio)
FIXED: Better error handling - always returns proper list
"""

from mcp.server.fastmcp import FastMCP
from typing import List, Dict
import json
import sys

# Redirect stdout to stderr during imports
_original_stdout = sys.stdout
sys.stdout = sys.stderr

try:
    from document_loader import DocumentLoader
finally:
    sys.stdout = _original_stdout

# Initialize FastMCP server
mcp = FastMCP("document-loader")

# Initialize document loader (singleton)
sys.stdout = sys.stderr
document_loader = DocumentLoader()
sys.stdout = _original_stdout


# ============================================================================
# MCP TOOLS
# ============================================================================

@mcp.tool()
def load_file(file_path: str) -> Dict:
    """
    Load a single document file.
    
    Args:
        file_path: Path to the file to load
        
    Returns:
        Document dictionary with id, text, and metadata
    """
    try:
        # Redirect stdout during operation
        _stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            result = document_loader.load_file(file_path)
        finally:
            sys.stdout = _stdout
        return result
    except Exception as e:
        # Return error in dict format
        sys.stderr.write(f"Error loading file {file_path}: {str(e)}\n")
        return {
            "error": str(e),
            "file_path": file_path
        }


@mcp.tool()
def load_directory(path: str, recursive: bool = True) -> List[Dict]:
    """
    Load all supported documents from a directory.
    
    Args:
        path: Directory path to load from
        recursive: Whether to search subdirectories (default: True)
        
    Returns:
        List of document dictionaries (always a list, even if empty or error)
    """
    try:
        # Redirect stdout during operation
        _stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            result = document_loader.load_directory(path, recursive)
        finally:
            sys.stdout = _stdout
        
        # Ensure we return a list
        if not isinstance(result, list):
            sys.stderr.write(f"Warning: load_directory returned {type(result)}, wrapping in list\n")
            return [result] if result else []
        
        return result
        
    except Exception as e:
        # ALWAYS return a list, even on error
        error_msg = f"Error loading directory {path}: {str(e)}"
        sys.stderr.write(error_msg + "\n")
        # Return empty list on error so the caller can continue
        return []


@mcp.tool()
def get_supported_extensions() -> List[str]:
    """
    Get list of supported file extensions.
    
    Returns:
        List of supported file extensions
    """
    return document_loader.supported_extensions


@mcp.tool()
def create_document(doc_id: str, text: str, metadata: dict = None) -> Dict:
    """
    Create a document dictionary manually.
    
    Args:
        doc_id: Unique document ID
        text: Document text
        metadata: Optional metadata dictionary
        
    Returns:
        Document dictionary
    """
    if metadata is None:
        metadata = {"type": "manual"}
    
    return document_loader.create_document(doc_id, text, metadata)


# ============================================================================
# RESOURCES (Optional - for exposing loader info)
# ============================================================================

@mcp.resource("loader://supported-formats")
def supported_formats_resource() -> str:
    """Expose supported file formats as a resource"""
    formats = {
        "extensions": document_loader.supported_extensions,
        "description": "Supported document formats for loading"
    }
    return json.dumps(formats, indent=2)


# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == "__main__":
    # Ensure all output goes to stderr except JSON-RPC
    import warnings
    warnings.filterwarnings("ignore")
    
    # Run the MCP server over stdio
    mcp.run(transport="stdio")