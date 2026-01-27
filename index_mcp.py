"""
Indexing Application - Offline RAG Index Builder (MCP Version)
Run this ONLY when documents change
Uses TRUE MCP over stdio
"""

from mcp_client import VectorStoreMCP, DocumentLoaderMCP
from config import DOCUMENTS_DIR

def main():
    print("=" * 70)
    print("📚 RAG INDEXER (MCP OVER STDIO)")
    print("Loads → Chunks → Embeds → Stores in ChromaDB via MCP")
    print("=" * 70)

    # MCP clients (communicating via stdio)
    print("\n🔌 Connecting to MCP servers...")
    vector_mcp = VectorStoreMCP()
    loader_mcp = DocumentLoaderMCP()
    print("✓ Connected to MCP servers")

    # --------------------------------------------------
    # Clear existing index (full rebuild)
    # --------------------------------------------------
    print("\n🧹 Clearing existing vector index...")
    vector_mcp.clear_all()

    # --------------------------------------------------
    # Load documents
    # --------------------------------------------------
    print("\n📄 Loading documents via MCP...")
    documents = loader_mcp.load_directory(
        str(DOCUMENTS_DIR),
        recursive=True
    )

    # Ensure documents is a list
    if not isinstance(documents, list):
        print(f"⚠️ Warning: documents is {type(documents)}, converting to list")
        if isinstance(documents, dict):
            # If it's a single document dict, wrap it in a list
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
    print("\n⚙️ Indexing documents via MCP...")
    result = vector_mcp.add_documents(documents)

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
    stats = vector_mcp.stats()
    
    if isinstance(stats, dict):
        for k, v in stats.items():
            print(f"  {k}: {v}")
    else:
        print(f"  Stats: {stats}")


if __name__ == "__main__":
    main()