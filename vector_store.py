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
        """Initialize ChromaDB and embedding model"""
        print(f"Initializing vector store at: {CHROMA_DB_PATH}")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        
        # Initialize embedding model
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Local RAG document store"}
        )
        
        print(f"✓ Vector store initialized with {self.collection.count()} documents")
    
    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, 
                   overlap: int = CHUNK_OVERLAP) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to chunk
            chunk_size: Size of each chunk in characters
            overlap: Number of overlapping characters between chunks
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end].strip()
            
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            start += chunk_size - overlap
        
        return chunks
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_tensor=False
        )
        return embeddings.tolist()
    
    def add_document(self, doc_id: str, text: str, metadata: Optional[Dict] = None) -> int:
        """
        Add a single document to the vector store
        
        Args:
            doc_id: Unique identifier for the document
            text: Document text content
            metadata: Optional metadata dictionary
            
        Returns:
            Number of chunks created
        """
        if not text or not text.strip():
            print(f"⚠ Warning: Empty document skipped: {doc_id}")
            return 0
        
        # Chunk the text
        chunks = self.chunk_text(text)
        
        if not chunks:
            print(f"⚠ Warning: No chunks created for: {doc_id}")
            return 0
        
        # Generate embeddings
        embeddings = self.embed_texts(chunks)
        
        # Prepare data for ChromaDB
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                **(metadata or {}),
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            for i in range(len(chunks))
        ]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )
        
        return len(chunks)
    
    def add_documents(self, documents: List[Dict]) -> Dict:
        """
        Add multiple documents to the vector store
        
        Args:
            documents: List of dicts with keys: 'id', 'text', 'metadata'
            
        Returns:
            Summary statistics
        """
        total_docs = len(documents)
        total_chunks = 0
        successful = 0
        failed = []
        
        for doc in documents:
            try:
                doc_id = doc['id']
                text = doc['text']
                metadata = doc.get('metadata', {})
                
                num_chunks = self.add_document(doc_id, text, metadata)
                
                if num_chunks > 0:
                    total_chunks += num_chunks
                    successful += 1
                    print(f"✓ Indexed: {doc_id} ({num_chunks} chunks)")
                else:
                    failed.append(doc_id)
                    
            except Exception as e:
                failed.append(doc.get('id', 'unknown'))
                print(f"✗ Error indexing {doc.get('id', 'unknown')}: {str(e)}")
        
        return {
            "total_documents": total_docs,
            "successful": successful,
            "failed": len(failed),
            "total_chunks": total_chunks,
            "failed_docs": failed
        }
    
    def query(self, query_text: str, n_results: int = N_RETRIEVAL_RESULTS) -> Dict:
        """
        Search for relevant documents
        
        Args:
            query_text: Search query
            n_results: Number of results to return
            
        Returns:
            Dictionary with documents, metadatas, and distances
        """
        if not query_text or not query_text.strip():
            return {"documents": [], "metadatas": [], "distances": []}
        
        # Generate query embedding
        query_embedding = self.embed_texts([query_text])[0]
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.collection.count())
        )
        
        # Flatten results (ChromaDB returns nested lists)
        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else []
        }
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete all chunks of a document
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if successful
        """
        try:
            # Get all chunk IDs for this document
            results = self.collection.get(
                where={"doc_id": doc_id}
            )
            
            if results and results["ids"]:
                self.collection.delete(ids=results["ids"])
                print(f"✓ Deleted document: {doc_id} ({len(results['ids'])} chunks)")
                return True
            else:
                print(f"⚠ Document not found: {doc_id}")
                return False
                
        except Exception as e:
            print(f"✗ Error deleting {doc_id}: {str(e)}")
            return False
    
    def clear_all(self) -> bool:
        """
        Delete all documents from the collection
        
        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "Local RAG document store"}
            )
            print("✓ All documents cleared")
            return True
        except Exception as e:
            print(f"✗ Error clearing documents: {str(e)}")
            return False
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the vector store
        
        Returns:
            Dictionary with statistics
        """
        count = self.collection.count()
        
        # Get unique document IDs
        if count > 0:
            results = self.collection.get()
            unique_docs = set()
            if results and results["metadatas"]:
                unique_docs = {m.get("doc_id") for m in results["metadatas"] if m.get("doc_id")}
        else:
            unique_docs = set()
        
        return {
            "total_chunks": count,
            "unique_documents": len(unique_docs),
            "collection_name": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL
        }


# ============================================================================
# Standalone Usage Example
# ============================================================================

if __name__ == "__main__":
    # Initialize vector store
    store = VectorStore()
    
    # Example: Add some documents
    sample_docs = [
        {
            "id": "python_intro",
            "text": "Python is a high-level programming language created by Guido van Rossum.",
            "metadata": {"topic": "programming", "language": "python"}
        },
        {
            "id": "ml_basics",
            "text": "Machine learning is a subset of AI that learns from data.",
            "metadata": {"topic": "AI", "subtopic": "machine_learning"}
        }
    ]
    
    # Add documents
    result = store.add_documents(sample_docs)
    print(f"\nIndexing result: {result}")
    
    # Query
    print("\n--- Testing Query ---")
    results = store.query("What is Python?", n_results=2)
    print(f"Found {len(results['documents'])} results")
    for i, doc in enumerate(results['documents']):
        print(f"\nResult {i+1}: {doc[:100]}...")
    
    # Get stats
    print("\n--- Vector Store Stats ---")
    stats = store.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")