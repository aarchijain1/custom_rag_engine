"""
Indexing Application - Offline RAG Index Builder
Run this ONLY when documents change
"""

from mcp.vector_store_mcp import VectorStoreMCP
from mcp.document_loader_mcp import DocumentLoaderMCP
from config import DOCUMENTS_DIR

def main():
    print("=" * 70)
    print("📚 RAG INDEXER")
    print("Loads → Chunks → Embeds → Stores in ChromaDB")
    print("=" * 70)

    # MCP resources
    vector_mcp = VectorStoreMCP()
    loader_mcp = DocumentLoaderMCP()

    # --------------------------------------------------
    # Clear existing index (full rebuild)
    # --------------------------------------------------
    print("\n🧹 Clearing existing vector index...")
    vector_mcp.clear_all()

    # --------------------------------------------------
    # Load documents
    # --------------------------------------------------
    print("\n📄 Loading documents...")
    documents = loader_mcp.load_directory(
        str(DOCUMENTS_DIR),
        recursive=True
    )

    if not documents:
        print("⚠️ No documents found. Index not created.")
        return

    # --------------------------------------------------
    # Index documents
    # --------------------------------------------------
    print("\n⚙️ Indexing documents...")
    result = vector_mcp.add_documents(documents)

    print("\n✅ Indexing complete")
    print(f"Documents indexed : {result['successful']}")
    print(f"Total chunks      : {result['total_chunks']}")

    # --------------------------------------------------
    # Final stats
    # --------------------------------------------------
    stats = vector_mcp.stats()
    print("\n📊 Vector Store Stats")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
