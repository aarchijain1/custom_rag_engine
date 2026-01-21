"""
Main Application - Interactive RAG Agent Interface
Entry point for the Local RAG Agent with LangGraph
"""

import sys
from pathlib import Path

from config import DOCUMENTS_DIR, VERBOSE_MODE, SHOW_RETRIEVED_DOCS
from document_loader import DocumentLoader, create_sample_documents
from vector_store import VectorStore
from agent_graph import RAGAgent


def print_header():
    """Print application header"""
    print("=" * 70)
    print("LOCAL RAG AGENT WITH LANGGRAPH")
    print("Powered by: Gemini API + ChromaDB + Sentence Transformers")
    print("=" * 70)


def print_menu():
    """Print main menu options"""
    print("\n" + "=" * 70)
    print("MENU OPTIONS")
    print("=" * 70)
    print("1. Chat with agent")
    print("2. Index documents from directory")
    print("3. Index a single document")
    print("4. View vector store statistics")
    print("5. Clear all indexed documents")
    print("6. Create sample documents")
    print("7. Settings")
    print("0. Exit")
    print("=" * 70)


def chat_mode(agent: RAGAgent):
    """
    Interactive chat mode
    
    Args:
        agent: RAG agent instance
    """
    print("\n" + "=" * 70)
    print("CHAT MODE")
    print("Type your questions below. Commands:")
    print("  'quit' or 'exit' - Return to main menu")
    print("  'stats' - Show vector store statistics")
    print("=" * 70 + "\n")
    
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Returning to main menu...")
                break
            
            if user_input.lower() == 'stats':
                stats = agent.get_stats()
                print("\n📊 Vector Store Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue
            
            # Query the agent
            print("\n🤖 Assistant: ", end="", flush=True)
            result = agent.query(user_input)
            
            # Print answer
            print(result["answer"])
            
            # Show metadata if verbose
            if VERBOSE_MODE or SHOW_RETRIEVED_DOCS:
                if result.get("used_rag"):
                    print(f"\n💡 [Used RAG - Retrieved {result['metadata'].get('num_documents', 0)} chunks]")
                else:
                    print("\n💡 [Direct answer - no RAG]")
            
            # Show error if any
            if result.get("error"):
                print(f"\n⚠️  Error: {result['error']}")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Returning to main menu...")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


def index_directory(vector_store: VectorStore):
    """
    Index all documents from a directory
    
    Args:
        vector_store: Vector store instance
    """
    print("\n" + "=" * 70)
    print("INDEX DIRECTORY")
    print("=" * 70)
    
    # Get directory path
    dir_path = input(f"\nEnter directory path (default: {DOCUMENTS_DIR}): ").strip()
    if not dir_path:
        dir_path = str(DOCUMENTS_DIR)
    
    # Ask about recursive search
    recursive_input = input("Search subdirectories? (y/n, default: y): ").strip().lower()
    recursive = recursive_input != 'n'
    
    try:
        print(f"\n📂 Loading documents from: {dir_path}")
        print(f"   Recursive: {recursive}\n")
        
        # Load documents
        loader = DocumentLoader()
        documents = loader.load_directory(dir_path, recursive=recursive)
        
        if not documents:
            print("\n⚠️  No documents found!")
            return
        
        # Index documents
        print(f"\n📊 Indexing {len(documents)} documents...")
        result = vector_store.add_documents(documents)
        
        # Show results
        print("\n" + "=" * 70)
        print("INDEXING RESULTS")
        print("=" * 70)
        print(f"✓ Total documents: {result['total_documents']}")
        print(f"✓ Successfully indexed: {result['successful']}")
        print(f"✓ Total chunks created: {result['total_chunks']}")
        if result['failed'] > 0:
            print(f"✗ Failed: {result['failed']}")
            print(f"  Failed documents: {', '.join(result['failed_docs'])}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


def index_single_document(vector_store: VectorStore):
    """
    Index a single document file
    
    Args:
        vector_store: Vector store instance
    """
    print("\n" + "=" * 70)
    print("INDEX SINGLE DOCUMENT")
    print("=" * 70)
    
    file_path = input("\nEnter file path: ").strip()
    
    if not file_path:
        print("⚠️  No file path provided")
        return
    
    try:
        print(f"\n📄 Loading document: {file_path}")
        
        # Load document
        loader = DocumentLoader()
        document = loader.load_file(file_path)
        
        # Index document
        print(f"📊 Indexing document...")
        num_chunks = vector_store.add_document(
            doc_id=document['id'],
            text=document['text'],
            metadata=document['metadata']
        )
        
        print(f"\n✓ Successfully indexed: {document['id']}")
        print(f"✓ Created {num_chunks} chunks")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


def view_stats(vector_store: VectorStore):
    """
    Display vector store statistics
    
    Args:
        vector_store: Vector store instance
    """
    print("\n" + "=" * 70)
    print("VECTOR STORE STATISTICS")
    print("=" * 70)
    
    stats = vector_store.get_stats()
    
    print(f"\n📊 Collection: {stats['collection_name']}")
    print(f"📚 Unique documents: {stats['unique_documents']}")
    print(f"🧩 Total chunks: {stats['total_chunks']}")
    print(f"🔢 Embedding model: {stats['embedding_model']}")
    print("\n" + "=" * 70)


def clear_documents(vector_store: VectorStore):
    """
    Clear all documents from vector store
    
    Args:
        vector_store: Vector store instance
    """
    print("\n" + "=" * 70)
    print("CLEAR ALL DOCUMENTS")
    print("=" * 70)
    
    confirm = input("\n⚠️  Are you sure? This will delete ALL indexed documents. (yes/no): ").strip().lower()
    
    if confirm == 'yes':
        print("\n🗑️  Clearing all documents...")
        success = vector_store.clear_all()
        
        if success:
            print("✓ All documents cleared successfully")
        else:
            print("❌ Failed to clear documents")
    else:
        print("❌ Operation cancelled")


def create_samples():
    """Create sample documents"""
    print("\n" + "=" * 70)
    print("CREATE SAMPLE DOCUMENTS")
    print("=" * 70)
    
    output_dir = input(f"\nEnter output directory (default: {DOCUMENTS_DIR}): ").strip()
    if not output_dir:
        output_dir = str(DOCUMENTS_DIR)
    
    try:
        create_sample_documents(output_dir)
        print(f"\n✓ Sample documents created successfully!")
        print(f"  You can now index them using option 2")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


def settings_menu():
    """Display and modify settings"""
    print("\n" + "=" * 70)
    print("SETTINGS")
    print("=" * 70)
    print(f"\n1. Verbose mode: {VERBOSE_MODE}")
    print(f"2. Show retrieved documents: {SHOW_RETRIEVED_DOCS}")
    print(f"3. Documents directory: {DOCUMENTS_DIR}")
    print("\nNote: Restart required for some settings changes")
    print("Press Enter to return to main menu...")
    input()


def main():
    """Main application loop"""
    print_header()
    
    try:
        # Initialize components
        print("\n🔧 Initializing RAG Agent...")
        agent = RAGAgent()
        vector_store = VectorStore()
        
        # Check if documents exist
        stats = vector_store.get_stats()
        if stats['unique_documents'] == 0:
            print("\n⚠️  No documents indexed yet!")
            print("💡 Tip: Use option 6 to create sample documents, then option 2 to index them")
        
        # Main loop
        while True:
            print_menu()
            
            try:
                choice = input("\nSelect option (0-7): ").strip()
                
                if choice == '0':
                    print("\n👋 Goodbye!")
                    sys.exit(0)
                
                elif choice == '1':
                    chat_mode(agent)
                
                elif choice == '2':
                    index_directory(vector_store)
                
                elif choice == '3':
                    index_single_document(vector_store)
                
                elif choice == '4':
                    view_stats(vector_store)
                
                elif choice == '5':
                    clear_documents(vector_store)
                
                elif choice == '6':
                    create_samples()
                
                elif choice == '7':
                    settings_menu()
                
                else:
                    print("\n❌ Invalid option. Please try again.")
            
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted. Returning to menu...")
                continue
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()