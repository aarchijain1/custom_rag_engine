"""
Main Application - Simple Clean Version
Chat-Only RAG Agent with simple MCP management
"""

from agent_graph import MultiAgentRAGSystem as RAGAgent
from config import DOCUMENTS_DIR
from mcp_manager import get_vector_stats


def main():
    print("=" * 70)
    print("🤖 LOCAL RAG AGENT (HTTP MCP)")
    print("Gemini + ChromaDB + LangGraph + HTTP MCP")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Ensure index exists
    # ------------------------------------------------------------------
    print("\n📚 Checking vector index...")
    stats = get_vector_stats()

    if stats.get("total_chunks", 0) == 0:
        raise RuntimeError(
            "❌ No index found.\n"
            "Run `python index_simple.py` before starting the chat assistant."
        )

    print(f"✓ Loaded index ({stats['total_chunks']} chunks) via HTTP MCP")

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
