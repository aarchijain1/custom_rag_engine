"""
Main Application - Chat-Only RAG Agent (HTTP MCP Version)
All resources accessed via HTTP MCP with auto-start
"""

from agent_graph import MultiAgentRAGSystem as RAGAgent
from mcp_client import VectorStoreMCP
from config import DOCUMENTS_DIR, MCP_SERVER_URL, MCP_AUTO_START


def main():
    print("=" * 70)
    print("🤖 LOCAL RAG AGENT (HTTP MCP)")
    print("Gemini + ChromaDB + LangGraph + HTTP MCP")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Connect to MCP server (auto-starts if needed)
    # ------------------------------------------------------------------
    print("\n🔌 Connecting to MCP server...")
    vector_mcp = VectorStoreMCP(
        server_url=MCP_SERVER_URL,
        auto_start_server=MCP_AUTO_START
    )
    print("✓ Connected to MCP server")

    # ------------------------------------------------------------------
    # Ensure index exists
    # ------------------------------------------------------------------
    stats = vector_mcp.stats()

    if stats.get("total_chunks", 0) == 0:
        raise RuntimeError(
            "❌ No index found.\n"
            "Run `python index.py` before starting the chat assistant."
        )

    print(f"\n📚 Loaded index ({stats['total_chunks']} chunks) via HTTP MCP")

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
            print("\n💡 MCP server is still running.")
            print("   Use 'python mcp_manage.py stop' to stop it.")
            break

        except Exception as e:
            print(f"\n⚠️ Error: {str(e)}")


if __name__ == "__main__":
    main()