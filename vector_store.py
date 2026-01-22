"""
Vector Store Management using ChromaDB
Handles document embedding and retrieval
"""

import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    N_RETRIEVAL_RESULTS
)


class VectorStore:
    """Manages vector database operations"""

    def __init__(self):
        print(f"Initializing vector store at: {CHROMA_DB_PATH}")

        # ChromaDB client
        self.client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

        # Embedding model
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        # Collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Local RAG document store"}
        )

        print(f"✓ Vector store initialized with {self.collection.count()} chunks")

    # ------------------------------------------------------------------
    # TEXT CHUNKING
    # ------------------------------------------------------------------

    def chunk_text(
        self,
        text: str,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP
    ) -> List[str]:

        if not text or not text.strip():
            return []

        chunks = []
        start = 0
        length = len(text)

        while start < length:
            end = start + chunk_size
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            start += chunk_size - overlap

        return chunks

    # ------------------------------------------------------------------
    # EMBEDDINGS
    # ------------------------------------------------------------------

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_tensor=False
        )
        return embeddings.tolist()

    # ------------------------------------------------------------------
    # ADD DOCUMENTS
    # ------------------------------------------------------------------

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict] = None
    ) -> int:

        if not text or not text.strip():
            print(f"⚠ Skipping empty document: {doc_id}")
            return 0

        chunks = self.chunk_text(text)
        if not chunks:
            return 0

        embeddings = self.embed_texts(chunks)

        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

        metadatas = [
            {
                **(metadata or {}),
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            for i in range(len(chunks))
        ]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        return len(chunks)

    def add_documents(self, documents: List[Dict]) -> Dict:
        total_chunks = 0
        success = 0
        failed = []

        for doc in documents:
            try:
                chunks = self.add_document(
                    doc_id=doc["id"],
                    text=doc["text"],
                    metadata=doc.get("metadata", {})
                )

                if chunks:
                    total_chunks += chunks
                    success += 1
                    print(f"✓ Indexed: {doc['id']} ({chunks} chunks)")
                else:
                    failed.append(doc["id"])

            except Exception as e:
                failed.append(doc.get("id", "unknown"))
                print(f"✗ Failed indexing {doc.get('id')}: {e}")

        return {
            "successful": success,
            "failed": len(failed),
            "total_chunks": total_chunks,
            "failed_docs": failed,
        }

    # ------------------------------------------------------------------
    # QUERY
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        n_results: int = N_RETRIEVAL_RESULTS
    ) -> List[Dict]:

        if not query_text or not query_text.strip():
            return []

        query_embedding = self.embed_texts([query_text])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.collection.count())
        )

        docs = []

        if not results or not results.get("documents"):
            return docs

        for i in range(len(results["documents"][0])):
            docs.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })

        return docs

    # ------------------------------------------------------------------
    # DELETE / CLEAR
    # ------------------------------------------------------------------

    def delete_document(self, doc_id: str) -> bool:
        try:
            results = self.collection.get(where={"doc_id": doc_id})
            if results and results["ids"]:
                self.collection.delete(ids=results["ids"])
                print(f"✓ Deleted document: {doc_id}")
                return True
            return False
        except Exception as e:
            print(f"✗ Delete failed: {e}")
            return False

    def clear_all(self) -> bool:
        try:
            self.client.delete_collection(COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "Local RAG document store"}
            )
            print("✓ Vector store cleared")
            return True
        except Exception as e:
            print(f"✗ Clear failed: {e}")
            return False

    # ------------------------------------------------------------------
    # STATS
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        count = self.collection.count()

        unique_docs = set()
        if count > 0:
            results = self.collection.get()
            for m in results.get("metadatas", []):
                if m.get("doc_id"):
                    unique_docs.add(m["doc_id"])

        return {
            "total_chunks": count,
            "unique_documents": len(unique_docs),
            "collection_name": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL,
        }


# ------------------------------------------------------------------
# DEBUG
# ------------------------------------------------------------------

if __name__ == "__main__":
    store = VectorStore()
    print(store.get_stats())
