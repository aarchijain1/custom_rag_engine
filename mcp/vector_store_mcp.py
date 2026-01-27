"""
MCP Wrapper for Vector Store
All vector DB access MUST go through this layer
"""

from typing import List, Dict, Any
from vector_store import VectorStore


class VectorStoreMCP:
    """
    MCP-style gateway for Vector DB
    """

    def __init__(self):
        self._store = VectorStore()

    # ---------------------------
    # MCP exposed capabilities
    # ---------------------------

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        return self._store.search(query, k)

    def add_documents(self, documents: List[Dict]) -> Dict:
        return self._store.add_documents(documents)

    def add_document(self, doc_id: str, text: str, metadata: Dict):
        return self._store.add_document(doc_id, text, metadata)

    def clear_all(self) -> bool:
        return self._store.clear_all()

    def stats(self) -> Dict:
        return self._store.get_stats()
