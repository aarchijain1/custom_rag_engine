# Local RAG Agent with LangGraph

A fully local Retrieval Augmented Generation (RAG) agent built with LangGraph, Gemini API, and ChromaDB. No Google Cloud Platform resources required!

## 🎯 Features

- **🤖 Multi-Agent Architecture**: Master agent orchestrates specialized sub-agents
- **📚 RAG System**: Automatic document retrieval from local vector database
- **💬 Natural Conversation**: User-friendly chat interface
- **🔍 Smart Routing**: Automatically decides when to use RAG vs direct answers
- **📄 Multi-Format Support**: TXT, PDF, DOCX, JSON, Markdown
- **💾 Persistent Storage**: Local ChromaDB vector database
- **⚡ Offline Indexing**: Pre-build embeddings before runtime with `index.py`
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
custom_rag_engine/
├── .env                    # API keys (create this)
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── config.py              # Configuration settings
├── index.py               # Offline document indexing
├── main.py                # Runtime chat interface
├── vector_store.py        # ChromaDB vector database manager
├── document_loader.py     # Multi-format document loader
├── agent_nodes.py         # LangGraph agent nodes
├── agent_graph.py         # LangGraph workflow builder
│
├── mcp/                   # MCP server (optional)
│   └── vector_store_mcp.py
│
├── docs/                  # Your documents (auto-created)
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
cd custom_rag_engine

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

### 4. Index Your Documents (Required First Step)

**Before you can chat, you must index your documents:**

```bash
# Place your documents in the docs/ folder
# Then run the indexing script
python index.py
```

This will:
- Load all documents from the `docs/` folder
- Split them into chunks
- Generate embeddings
- Store them in the ChromaDB vector database (`chroma_db/`)

**Note:** Indexing is a one-time step (or run again when you add new documents). The vector database persists between sessions.

### 5. Run the Chat Application

```bash
python main.py
```

The chat interface will use the pre-built vector database for fast retrieval.

## 📖 Usage Guide

### Workflow: Index → Chat

1. **Add Documents**: Place your files in the `docs/` folder
2. **Index Documents**: Run `python index.py` to build embeddings
3. **Chat**: Run `python main.py` to start asking questions

### Indexing (index.py)

The indexing script processes documents **offline** before runtime:

```bash
python index.py
```

**What it does:**
- Scans the `docs/` directory for TXT, PDF, DOCX, JSON, and Markdown files
- Chunks each document (default: 500 chars with 50 char overlap)
- Generates embeddings using Sentence Transformers
- Stores vectors in ChromaDB for fast retrieval

**When to re-run:**
- After adding new documents
- After modifying existing documents
- When changing chunk size or embedding model in `config.py`

### Chat Interface (main.py)

The runtime application provides an interactive chat interface:

```bash
python main.py
```

**Menu Options:**

```
1. Chat with agent           - Interactive Q&A
2. View statistics          - See indexed document stats
3. Settings                 - View current configuration
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

# Change chunk size (requires re-indexing)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Number of documents to retrieve
N_RETRIEVAL_RESULTS = 3

# Enable verbose mode for debugging
VERBOSE_MODE = True
```

**Important:** If you change `CHUNK_SIZE`, `CHUNK_OVERLAP`, or `EMBEDDING_MODEL`, you must re-run `python index.py` to rebuild the vector database.

### Change Embedding Model (config.py)

```python
# Faster, smaller model (default)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Better quality, slower
EMBEDDING_MODEL = "all-mpnet-base-v2"
```

**Remember to re-index after changing the embedding model!**

### Custom Prompts (config.py)

Modify `CLASSIFICATION_PROMPT`, `RAG_ANSWER_PROMPT`, and `DIRECT_ANSWER_PROMPT` to customize agent behavior.

## 📚 Adding Your Own Documents

### Method 1: Using the Interface

1. Copy your documents to the `docs/` folder
2. Run `python index.py` to index them
3. Run `python main.py` to start chatting

### Method 2: Programmatically

```python
from document_loader import DocumentLoader
from vector_store import VectorStore

# Initialize
loader = DocumentLoader()
vector_store = VectorStore()

# Load and index
documents = loader.load_directory("./docs")
for doc in documents:
    vector_store.add_document(doc['id'], doc['text'], doc['metadata'])
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

### Issue: "No documents found" when chatting
**Solution:** 
1. Verify documents are in the `docs/` folder
2. Run `python index.py` to index your documents
3. Check that file extensions are supported (.txt, .pdf, .docx, .json, .md)

### Issue: "Vector database is empty"
**Solution:** You forgot to run the indexing step!
```bash
python index.py
```

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

### Issue: Outdated embeddings after adding documents
**Solution:** Re-run the indexing script
```bash
python index.py
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
   - **Remember to re-index after changing!**

3. **Pre-index documents**:
   - Run `index.py` once, then reuse the database across sessions
   - Much faster than indexing at runtime

4. **Batch document indexing**:
   - Place all documents in `docs/` folder before running `index.py`
   - More efficient than indexing one-by-one

## 🚀 Next Steps

### Enhancements You Can Add

1. **Web Search Integration**: Add web search alongside RAG
2. **Conversation Memory**: Track chat history across sessions
3. **Multi-Query RAG**: Generate multiple search queries for better retrieval
4. **Re-ranking**: Add a re-ranker to improve retrieval quality
5. **Streaming Responses**: Stream answers token-by-token
6. **Web Interface**: Build a Streamlit or Gradio UI
7. **API Server**: Create FastAPI endpoints
8. **Incremental Indexing**: Add new documents without rebuilding entire database

### Example: Incremental Indexing

```python
# Add to index.py
def index_new_documents(doc_paths):
    """Index only specific new documents"""
    loader = DocumentLoader()
    vector_store = VectorStore()
    
    for path in doc_paths:
        doc = loader.load_file(path)
        vector_store.add_document(doc['id'], doc['text'], doc['metadata'])
```

## 📝 File Descriptions

| File | Purpose |
|------|---------|
| `config.py` | All configuration settings and prompts |
| `index.py` | **Offline document indexing and embedding generation** |
| `main.py` | **Runtime chat interface and agent execution** |
| `vector_store.py` | ChromaDB vector database management |
| `document_loader.py` | Load TXT, PDF, DOCX, JSON, MD files |
| `agent_nodes.py` | Individual agent functions (nodes) |
| `agent_graph.py` | LangGraph workflow and routing |
| `mcp/vector_store_mcp.py` | MCP server (optional) |

## 🔄 Typical Workflow

```
1. Add documents → docs/
2. Run: python index.py (builds vector DB)
3. Run: python main.py (start chatting)
4. Add more documents → Re-run index.py
```

## 🤝 Contributing

Feel free to extend this project! Some ideas:
- Add support for more file formats
- Implement advanced RAG techniques
- Create a web UI
- Add more agent types
- Build incremental indexing

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
2. Verify you've run `python index.py` before chatting
3. Review the example usage
4. Check configuration settings

---

**Happy RAG-ing! 🚀**