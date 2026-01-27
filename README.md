# RAG Agent with FastMCP

A production-ready Retrieval Augmented Generation (RAG) agent built with **FastMCP**, LangGraph, Google Gemini, and ChromaDB. Uses **True Model Context Protocol** for process isolation and scalability.

## 🎯 Features

- **🔌 FastMCP Servers**: Clean `@mcp.tool()` decorators for MCP protocol
- **📡 True MCP Protocol**: JSON-RPC 2.0 over stdio transport
- **🔄 Process Isolation**: MCP servers run as independent processes
- **🤖 Multi-Agent Architecture**: Master agent orchestrates retrieval and answering
- **📚 RAG System**: Semantic search with ChromaDB vector database
- **💬 Natural Conversation**: User-friendly chat interface
- **🔍 Smart Routing**: LLM decides when to use RAG vs direct answers
- **📄 Multi-Format Support**: TXT, PDF, DOCX, JSON, Markdown
- **💾 Persistent Storage**: Local ChromaDB with 90 indexed chunks
- **⚡ Offline Indexing**: Pre-build embeddings with `index_mcp.py`
- **🔧 Fully Configurable**: Easy customization via `config.py`
- **🐛 MCP Inspector**: Debug with official MCP tools
- **🌐 No Cloud Required**: Runs locally (except Gemini API calls)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   USER QUERY                            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │   RAG AGENT          │
          │  (LangGraph)         │
          │  - Master Router     │
          │  - Retrieval         │
          │  - RAG Answer        │
          └──────────┬───────────┘
                     │
            ┌────────▼────────┐
            │  MCP Client     │ (Sync Wrappers)
            └────────┬────────┘
                     │ stdio (JSON-RPC 2.0)
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────────┐    ┌──────────────────┐
│ FastMCP Server    │    │ FastMCP Server   │
│ Vector Store      │    │ Doc Loader       │
│                   │    │                  │
│ @mcp.tool()       │    │ @mcp.tool()      │
│ - search_docs()   │    │ - load_dir()     │
│ - add_docs()      │    │ - load_file()    │
│ - clear()         │    │                  │
└────────┬──────────┘    └──────────────────┘
         │
         ▼
┌─────────────────┐
│   ChromaDB      │ (90 chunks from 6 documents)
│   Embeddings    │
└─────────────────┘
```

## 📁 Project Structure

```
local-rag-agent/
├── .env                          # API keys (create this)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── config.py                     # Configuration settings
├── index_mcp.py                  # Offline indexing (via MCP)
├── main_mcp.py                   # Runtime chat (via MCP)
│
├── mcp_server_vector.py          # FastMCP: Vector store server
├── mcp_server_documents.py       # FastMCP: Document loader server
├── mcp_client.py                 # MCP client with sync wrappers
│
├── agent_nodes_mcp.py            # LangGraph nodes (uses MCP)
├── agent_graph.py                # LangGraph workflow
├── vector_store.py               # ChromaDB implementation
├── document_loader.py            # Multi-format document loader
├── llm.py                        # Gemini LLM setup
│
├── documents/                    # Your documents (6 PDFs + 1 JSON)
│   ├── TFS-lease-end-guide.pdf
│   ├── faq_data.json
│   └── ...
│
└── chroma_db/                    # Vector database (auto-created)
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9+
- Google Gemini API key (free from [Google AI Studio](https://aistudio.google.com/app/apikey))

### 2. Installation

```bash
# Clone or download this project
cd local-rag-agent

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install MCP SDK
pip install "mcp[cli]"
```

### 3. Configuration

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 4. Index Your Documents (Required First Step)

**Before you can chat, you must index your documents:**

```bash
# Documents are already in documents/ folder
# Run the indexing script via MCP
python index_mcp.py
```

Expected output:
```
📚 RAG INDEXER (MCP OVER STDIO)
🔌 Connecting to MCP servers...
✓ Connected to MCP servers
✓ Loaded 6 documents
⚙️ Indexing documents via MCP...
✅ Indexing complete
Documents indexed : 6
Total chunks      : 90
```

This will:
- Connect to FastMCP servers via stdio
- Load documents via document loader MCP server
- Send documents to vector store MCP server
- Generate embeddings and store in ChromaDB

**Note:** Indexing is a one-time step (or run again when adding documents).

### 5. Run the Chat Application

```bash
python main_mcp.py
```

Expected output:
```
🤖 LOCAL RAG AGENT (MCP OVER STDIO)
✓ Connected to MCP servers
📚 Loaded index (90 chunks) via MCP
✓ RAG Agent ready
💬 CHAT STARTED

🧑 User: 
```

## 📖 Usage Guide

### Workflow: Index → Chat

1. **Documents Provided**
2. **Index Documents**: Run `python index_mcp.py` 
3. **Chat**: Run `python main_mcp.py` to ask questions

### Indexing (index_mcp.py)

The indexing script uses **FastMCP** to process documents:

```bash
python index_mcp.py
```

**How it works:**
1. Connects to vector store MCP server
2. User asks question
3. Master router decides: RAG or direct answer?
4. If RAG: Calls `search_documents()` tool via MCP
5. Vector store server returns top 3 chunks
6. Gemini generates answer with context
7. Returns answer with sources

## 📝 File Descriptions

| File | Purpose |
|------|---------|
| **MCP Servers** | |
| `mcp_server_vector.py` | FastMCP server for vector operations |
| `mcp_server_documents.py` | FastMCP server for document loading |
| `mcp_client.py` | MCP client with sync/async wrappers |
| **Application** | |
| `agent_nodes_mcp.py` | LangGraph nodes (uses MCP client) |
| `agent_graph_mcp.py` | LangGraph workflow builder |
| `index_mcp.py` | Offline indexing via MCP |
| `main_mcp.py` | Runtime chat interface via MCP |
| **Core Logic** | |
| `vector_store.py` | ChromaDB implementation |
| `document_loader.py` | Multi-format file loader |
| `llm.py` | Gemini LLM setup |
| `config.py` | All configuration settings |


## 🎉 Success Checklist

Your system is working if:
- ✅ `python index_mcp.py` loads all 6 documents
- ✅ Shows "90 chunks" in stats
- ✅ `python main_mcp.py` starts without errors
- ✅ Agent answers questions about Toyota Financial Services
- ✅ Sources are included in responses
- ✅ Direct questions work without RAG
