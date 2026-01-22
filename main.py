from vector_store import VectorStore, auto_index_documents
from agent_graph import RAGAgent

def main():
    print("=" * 70)
    print("LOCAL RAG AGENT (CHAT MODE)")
    print("Powered by: Gemini + ChromaDB + LangGraph")
    print("=" * 70)

    # Vector store
    store = VectorStore()
    auto_index_documents(store)

    # Agent
    print("\n🔧 Initializing RAG Agent...")
    agent = RAGAgent()

    print("\n💬 Ask questions (type 'exit' to quit)\n")

    while True:
        q = input("🧑 You: ").strip()
        if q.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        answer = agent.chat(q)
        print("\n🤖 Assistant:")
        print(answer)
        print()


if __name__ == "__main__":
    main()
