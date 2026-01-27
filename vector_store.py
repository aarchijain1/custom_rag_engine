"""
Vector Store
Pure ChromaDB + SentenceTransformers
NO LangChain
"""

import os
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ---------------- #

CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "rag_documents"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
N_RETRIEVAL_RESULTS = 3

# ---------------------------------------- #


class VectorStore:
    def __init__(self):
        print(f"Initializing vector store at: {os.path.abspath(CHROMA_DIR)}")
        print(f"Loading embedding model: {EMBEDDING_MODEL}")

        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )

        print(f"✓ Vector store ready with {self.collection.count()} chunks")

    # ---------------- Utility ---------------- #

    def _embed(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_model.encode(
            texts,
            show_progress_bar=False
        ).tolist()

    def _chunk_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        length = len(text)

        while start < length:
            end = start + CHUNK_SIZE
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += CHUNK_SIZE - CHUNK_OVERLAP

        return chunks

    # ---------------- Indexing ---------------- #

    def add_document(self, doc_id: str, text: str, metadata: Dict):
        chunks = self._chunk_text(text)
        embeddings = self._embed(chunks)

        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = [{**metadata, "doc_id": doc_id} for _ in chunks]

        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )

    def add_documents(self, documents: List[Dict]) -> Dict:
        successful = 0
        total_chunks = 0

        for doc in documents:
            try:
                self.add_document(
                    doc_id=doc["id"],
                    text=doc["text"],
                    metadata=doc.get("metadata", {})
                )
                acknowledged_chunks = len(self._chunk_text(doc["text"]))
                successful += 1
                total_chunks += acknowledged_chunks
            except Exception as e:
                print(f"⚠ Failed to index {doc['id']}: {e}")

        return {
            "successful": successful,
            "total_chunks": total_chunks
        }

    # ---------------- Clear / Reset ---------------- #

    def clear_all(self) -> bool:
        """
        Completely reset the vector store by deleting and recreating the collection.
        This is the ONLY safe way to clear ChromaDB.
        """
        try:
            self.client.delete_collection(name=COLLECTION_NAME)

            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME
            )

            print("✓ Vector store fully reset")
            return True

        except Exception as e:
            print(f"✗ Failed to reset vector store: {e}")
            return False

    # ---------------- Query ---------------- #

    def search(self, query: str, k: int = N_RETRIEVAL_RESULTS) -> List[Dict]:
        if self.collection.count() == 0:
            return []

        embedding = self._embed([query])[0]

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(k, self.collection.count()),
        )

        docs = []
        for i in range(len(results["documents"][0])):
            docs.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
            })

        return docs

    # ---------------- Stats ---------------- #

    def get_stats(self) -> Dict:
        return {
            "total_chunks": self.collection.count(),
            "collection": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL,
        }
