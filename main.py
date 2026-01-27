"""
Main Application - Chat-Only RAG Agent
Auto indexes documents on startup
All resources accessed via MCP
"""

from agent_graph import RAGAgent
from mcp.vector_store_mcp import VectorStoreMCP
from mcp.document_loader_mcp import DocumentLoaderMCP
from config import DOCUMENTS_DIR


def main():
    print("=" * 70)
    print("🤖 LOCAL RAG AGENT (CHAT MODE)")
    print("Gemini + ChromaDB + LangGraph + MCP")
    print("=" * 70)

    # ------------------------------------------------------------------
    # MCP Resources
    # ------------------------------------------------------------------
    vector_mcp = VectorStoreMCP()
    loader_mcp = DocumentLoaderMCP()

    # ------------------------------------------------------------------
    # Auto index documents (only if needed)
    # ------------------------------------------------------------------
# ------------------------------------------------------------------
# Ensure index exists
# ------------------------------------------------------------------
    stats = vector_mcp.stats()

    if stats["total_chunks"] == 0:
        raise RuntimeError(
            "❌ No index found.\n"
            "Run `python index.py` before starting the chat assistant."
        )

    print(f"\n📚 Loaded index ({stats['total_chunks']} chunks)")



    # ------------------------------------------------------------------
    # Initialize Agent
    # ------------------------------------------------------------------
    print("\n🔧 Initializing RAG Agent...")
    agent = RAGAgent()
    print("✓ RAG Agent ready")

    # ------------------------------------------------------------------
    # Chat Loop
    # ------------------------------------------------------------------
    print("\n" + "=" * 30)
    print("💬 CHAT STARTED")
    print("Type 'exit' or 'quit' to stop")
    print("=" * 30)

    while True:
        try:
            question = input("\n🧑 User: ").strip()

            if question.lower() in {"exit", "quit"}:
                print("\n👋 Goodbye!")
                break

            if not question:
                continue

            answer = agent.chat(question)

            print("\n🤖 Assistant:")
            print(answer)

        except KeyboardInterrupt:
            print("\n\n👋 Session ended")
            break

        except Exception as e:
            print(f"\n⚠️ Error: {str(e)}")


if __name__ == "__main__":
    main()
