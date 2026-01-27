# Local RAG Agent with LangGraph

A fully local Retrieval Augmented Generation (RAG) agent built with LangGraph, Gemini API, and ChromaDB. No Google Cloud Platform resources required!

## 🎯 Features

- **🤖 Multi-Agent Architecture**: Master agent orchestrates specialized sub-agents
- **📚 RAG System**: Automatic document retrieval from local vector database
- **💬 Natural Conversation**: User-friendly chat interface
- **🔍 Smart Routing**: Automatically decides when to use RAG vs direct answers
- **📄 Multi-Format Support**: TXT, PDF, DOCX, JSON, Markdown
- **💾 Persistent Storage**: Local ChromaDB vector database
- **🔧 Fully Configurable**: Easy customization via config.py
- **🌐 No Cloud Required**: Runs entirely on your machine (except Gemini API calls)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     USER QUERY                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │   MASTER AGENT       │
          │  (Query Classifier)  │
          └──────────┬───────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────┐         ┌──────────────┐
│ RAG Path      │         │ Direct Path  │
│               │         │              │
│ 1. Retrieval  │         │ 1. Direct    │
│    Agent      │         │    Answer    │
│    ↓          │         │    ↓         │
│ 2. RAG Answer │         │ 2. END       │
│    Generator  │         │              │
│    ↓          │         │              │
│ 3. END        │         │              │
└───────────────┘         └──────────────┘
```

## 📁 Project Structure

```
local-rag-agent/
├── .env                    # API keys (create this)
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── config.py              # Configuration settings
├── vector_store.py        # ChromaDB vector database manager
├── document_loader.py     # Multi-format document loader
├── agent_nodes.py         # LangGraph agent nodes
├── agent_graph.py         # LangGraph workflow builder
├── main.py                # Main application entry point
│
├── documents/             # Your documents (auto-created)
│   ├── *.txt
│   ├── *.pdf
│   └── *.docx
│
└── chroma_db/            # Vector database (auto-created)
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9+
- Google Gemini API key (free from [Google AI Studio](https://makersuite.google.com/app/apikey))

### 2. Installation

```bash
# Clone or download this project
cd local-rag-agent

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file in the project root:

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 4. Run the Application

```bash
python main.py
```

## 📖 Usage Guide

### First-Time Setup

1. **Create Sample Documents** (Option 6)
   - Generates example documents in the `documents/` folder
   - Good for testing the system

2. **Index Documents** (Option 2)
   - Index the sample documents or your own documents
   - The system will process and store them in ChromaDB

3. **Start Chatting** (Option 1)
   - Ask questions about your documents
   - The agent automatically decides whether to search documents

### Menu Options

```
1. Chat with agent           - Interactive Q&A
2. Index documents           - Load documents from directory
3. Index single document     - Load one specific file
4. View statistics          - See indexed document stats
5. Clear all documents      - Reset the vector database
6. Create sample documents  - Generate test documents
7. Settings                 - View current configuration
0. Exit                     - Close the application
```

### Example Chat Session

```
🧑 You: Hello!
🤖 Assistant: Hello! How can I help you today?
💡 [Direct answer - no RAG]

🧑 You: What does the document say about Python?
🤖 Assistant: According to the documents, Python is a high-level 
programming language created by Guido van Rossum and first released 
in 1991. It emphasizes code readability...
💡 [Used RAG - Retrieved 3 chunks]

🧑 You: What is 2+2?
🤖 Assistant: 2+2 equals 4.
💡 [Direct answer - no RAG]
```

## 🔧 Customization

### Modify Configuration (config.py)

```python
# Change the LLM model
GEMINI_MODEL = "gemini-1.5-flash"  # or "gemini-1.5-pro"

# Adjust temperature (0.0 = deterministic, 1.0 = creative)
TEMPERATURE = 0.7

# Change chunk size
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Number of documents to retrieve
N_RETRIEVAL_RESULTS = 3

# Enable verbose mode for debugging
VERBOSE_MODE = True
```

### Change Embedding Model (config.py)

```python
# Faster, smaller model (default)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Better quality, slower
EMBEDDING_MODEL = "all-mpnet-base-v2"
```

### Custom Prompts (config.py)

Modify `CLASSIFICATION_PROMPT`, `RAG_ANSWER_PROMPT`, and `DIRECT_ANSWER_PROMPT` to customize agent behavior.

## 📚 Adding Your Own Documents

### Method 1: Using the Interface

1. Copy your documents to the `documents/` folder
2. Run `python main.py`
3. Select option 2 (Index documents)
4. Enter the path or press Enter for default

### Method 2: Programmatically

```python
from document_loader import DocumentLoader
from vector_store import VectorStore

# Initialize
loader = DocumentLoader()
vector_store = VectorStore()

# Load and index
documents = loader.load_directory("./my_documents")
vector_store.add_documents(documents)
```

## 🛠️ Advanced Usage

### Custom Document Processing

```python
from document_loader import DocumentLoader

loader = DocumentLoader()

# Create a custom document
doc = loader.create_document(
    doc_id="custom_doc_1",
    text="Your custom text here...",
    metadata={"author": "John Doe", "topic": "AI"}
)

# Index it
from vector_store import VectorStore
vector_store = VectorStore()
vector_store.add_document(doc['id'], doc['text'], doc['metadata'])
```

### Using the Agent Programmatically

```python
from agent_graph import RAGAgent

# Initialize agent
agent = RAGAgent()

# Simple query
answer = agent.chat("What is machine learning?")
print(answer)

# Detailed query with metadata
result = agent.query("What is Python?", verbose=True)
print(f"Answer: {result['answer']}")
print(f"Used RAG: {result['used_rag']}")
print(f"Sources: {result['metadata'].get('sources', [])}")
```

## 🐛 Troubleshooting

### Issue: "Module not found: chromadb"
**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: "Invalid API key"
**Solution:** Check your `.env` file
```bash
# Make sure .env contains:
GOOGLE_API_KEY=your_actual_key_here
```

### Issue: "No documents found"
**Solution:** 
1. Check documents are in the `documents/` folder
2. Verify file extensions are supported (.txt, .pdf, .docx, .json, .md)
3. Use option 6 to create sample documents

### Issue: PDF/DOCX files not loading
**Solution:** Install optional dependencies
```bash
pip install pypdf python-docx
```

### Issue: Slow embedding generation
**Solution:** Use a smaller embedding model in `config.py`:
```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Faster
```

## 🔒 Privacy & Security

- **All data stays local**: Documents are stored in local ChromaDB
- **Only API calls**: Gemini API is used only for text generation
- **No cloud storage**: No data sent to GCP or other cloud services
- **API key security**: Store in `.env`, never commit to version control

## 📊 Performance Tips

1. **Use GPU for embeddings** (if available):
   ```python
   # In vector_store.py
   self.embedding_model = SentenceTransformer(EMBEDDING_MODEL, device='cuda')
   ```

2. **Optimize chunk size** for your documents:
   - Smaller chunks (300-500): Better precision
   - Larger chunks (800-1000): More context

3. **Batch document indexing**:
   - Index multiple documents at once using option 2
   - Faster than indexing one-by-one

## 🚀 Next Steps

### Enhancements You Can Add

1. **Web Search Integration**: Add web search alongside RAG
2. **Conversation Memory**: Track chat history across sessions
3. **Multi-Query RAG**: Generate multiple search queries for better retrieval
4. **Re-ranking**: Add a re-ranker to improve retrieval quality
5. **Streaming Responses**: Stream answers token-by-token
6. **Web Interface**: Build a Streamlit or Gradio UI
7. **API Server**: Create FastAPI endpoints

### Example: Adding Web Search

```python
# In agent_nodes.py, add a web search node
def web_search_node(state: AgentState) -> AgentState:
    query = state["user_query"]
    # Add your web search logic here
    return state
```

## 📝 File Descriptions

| File | Purpose |
|------|---------|
| `config.py` | All configuration settings and prompts |
| `vector_store.py` | ChromaDB vector database management |
| `document_loader.py` | Load TXT, PDF, DOCX, JSON, MD files |
| `agent_nodes.py` | Individual agent functions (nodes) |
| `agent_graph.py` | LangGraph workflow and routing |
| `main.py` | Interactive CLI interface |

## 🤝 Contributing

Feel free to extend this project! Some ideas:
- Add support for more file formats
- Implement advanced RAG techniques
- Create a web UI
- Add more agent types

## 📄 License

MIT License - Feel free to use and modify!

## 🙏 Acknowledgments

- **LangGraph**: For the agent framework
- **Google Gemini**: For the LLM API
- **ChromaDB**: For the vector database
- **Sentence Transformers**: For local embeddings

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the example usage
3. Check configuration settings

---

**Happy RAG-ing! 🚀**