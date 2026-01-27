"""
MCP Wrapper for Document Loader
All filesystem access MUST go through this layer
"""

from typing import List, Dict
from document_loader import DocumentLoader


class DocumentLoaderMCP:
    """
    MCP-style gateway for filesystem document access
    """

    def __init__(self):
        self._loader = DocumentLoader()

    # ---------------------------
    # MCP exposed capabilities
    # ---------------------------

    def load_directory(self, path: str, recursive: bool = True) -> List[Dict]:
        return self._loader.load_directory(path, recursive)

    def load_file(self, file_path: str) -> Dict:
        return self._loader.load_file(file_path)
