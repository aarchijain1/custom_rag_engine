"""
Main Application - Chat-Only RAG Agent (MCP Version)
All resources accessed via TRUE MCP over stdio
"""

from agent_graph import MultiAgentRAGSystem as RAGAgent
from mcp_client import VectorStoreMCP
from config import DOCUMENTS_DIR


def main():
    print("=" * 70)
    print("🤖 LOCAL RAG AGENT (MCP OVER STDIO)")
    print("Gemini + ChromaDB + LangGraph + TRUE MCP")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Connect to MCP server
    # ------------------------------------------------------------------
    print("\n🔌 Connecting to MCP servers...")
    vector_mcp = VectorStoreMCP()
    print("✓ Connected to MCP servers")

    # ------------------------------------------------------------------
    # Ensure index exists
    # ------------------------------------------------------------------
    stats = vector_mcp.stats()

    if stats.get("total_chunks", 0) == 0:
        raise RuntimeError(
            "❌ No index found.\n"
            "Run `python index_mcp.py` before starting the chat assistant."
        )

    print(f"\n📚 Loaded index ({stats['total_chunks']} chunks) via MCP")

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