"""
Vector Store with Auto-Reindexing
Pure ChromaDB + SentenceTransformers
NO LangChain
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

import chromadb
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ---------------- #

DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"
INDEX_STATE_FILE = ".index_state.json"

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
            texts, show_progress_bar=False
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

    def clear(self):
        self.collection.delete(where={})

    # ---------------- Query ---------------- #

    def search(self, query: str, k: int = N_RETRIEVAL_RESULTS) -> List[Dict]:
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


# ===================================================================
# AUTO-INDEXING (hash-based)
# ===================================================================

def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _load_index_state() -> dict:
    if os.path.exists(INDEX_STATE_FILE):
        with open(INDEX_STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_index_state(state: dict):
    with open(INDEX_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def auto_index_documents(store: VectorStore):
    os.makedirs(DOCS_DIR, exist_ok=True)

    previous = _load_index_state()
    current = {}

    changed = False

    for file in Path(DOCS_DIR).glob("*"):
        if file.is_file():
            file_hash = _file_hash(file)
            current[file.name] = file_hash
            if previous.get(file.name) != file_hash:
                changed = True

    if previous.keys() != current.keys():
        changed = True

    if not changed:
        print("✅ No document changes detected.")
        return

    print("📄 Documents changed — rebuilding index...")

    store.clear()

    for file in Path(DOCS_DIR).glob("*"):
        if not file.is_file():
            continue

        try:
            text = file.read_text(encoding="utf-8", errors="ignore")
            store.add_document(
                doc_id=file.stem,
                text=text,
                metadata={"source": file.name},
            )
        except Exception as e:
            print(f"⚠ Failed to index {file.name}: {e}")

    _save_index_state(current)
    print(f"✅ Index rebuilt ({store.collection.count()} chunks)")
