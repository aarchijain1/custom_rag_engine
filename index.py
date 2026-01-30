"""
Document Indexer - Simple Clean Version
Uses Simple MCP Manager for all operations
"""

from config import DOCUMENTS_DIR
from mcp_manager import (clear_vector_store, load_documents, index_documents, get_vector_stats)

def main():
    print("=" * 70)
    print("📚 RAG INDEXER (HTTP MCP)")
    print("Loads → Chunks → Embeds → Stores in ChromaDB via HTTP MCP")
    print("=" * 70)

    # --------------------------------------------------
    # Clear existing index (full rebuild)
    # --------------------------------------------------
    print("\n🧹 Clearing existing vector index...")
    clear_vector_store()

    # --------------------------------------------------
    # Load documents
    # --------------------------------------------------
    print("\n📄 Loading documents via MCP...")
    documents = load_documents(str(DOCUMENTS_DIR), recursive=True)

    # Ensure documents is a list
    if not isinstance(documents, list):
        print(f"⚠️ Warning: documents is {type(documents)}, converting to list")
        if isinstance(documents, dict):
            documents = [documents]
        else:
            print(f"❌ Error: Cannot convert {type(documents)} to list")
            return

    if not documents:
        print("⚠️ No documents found. Index not created.")
        return

    print(f"✓ Loaded {len(documents)} documents")
    
    # Debug: show document structure
    if documents:
        print(f"\nFirst document structure:")
        first_doc = documents[0]
        print(f"  Type: {type(first_doc)}")
        if isinstance(first_doc, dict):
            print(f"  Keys: {list(first_doc.keys())}")

    # --------------------------------------------------
    # Index documents
    # --------------------------------------------------
    print("\n⚙️ Indexing documents via HTTP MCP...")
    result = index_documents(documents)

    print("\n✅ Indexing complete")
    
    # Handle different result formats
    if isinstance(result, dict):
        if 'successful' in result:
            print(f"Documents indexed : {result['successful']}")
        if 'total_chunks' in result:
            print(f"Total chunks      : {result['total_chunks']}")
        if 'error' in result:
            print(f"⚠️ Error: {result['error']}")
    elif isinstance(result, str):
        if 'Error' in result or 'error' in result:
            print(f"⚠️ {result}")
        else:
            print(f"Result: {result}")
    else:
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")

    # --------------------------------------------------
    # Final stats
    # --------------------------------------------------
    print("\n📊 Vector Store Stats")
    stats = get_vector_stats()
    
    if isinstance(stats, dict):
        for k, v in stats.items():
            print(f"  {k}: {v}")
    else:
        print(f"  Stats: {stats}")
    
    print("\n✨ Indexing complete! MCP server will be stopped automatically.")


if __name__ == "__main__":
    main()
