"""
Configuration settings for the Local RAG Agent
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# API Configuration
# ============================================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# Anthropic Claude API configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_TEMPERATURE = 0.7
CLAUDE_MAX_TOKENS = 2048

# ============================================================================
# Model Configuration
# ============================================================================

# Gemini model settings
GEMINI_MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 2048

# Embedding model (runs locally)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ============================================================================
# MCP Server Configuration - CORRECTED
# ============================================================================

# Unified MCP Server settings
MCP_SERVER_HOST = "127.0.0.1"
MCP_SERVER_PORT = 8765
MCP_SERVER_URL = f"http://{MCP_SERVER_HOST}:{MCP_SERVER_PORT}"  # No /mcp suffix!

# Auto-start server when agents need it
MCP_AUTO_START = True

# Server script path
MCP_SERVER_SCRIPT = "mcp_server_unified.py"

# ============================================================================
# Vector Database Configuration
# ============================================================================

# ChromaDB settings
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "rag_documents"

# ============================================================================
# RAG Configuration
# ============================================================================

# Text chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval settings
N_RETRIEVAL_RESULTS = 3

# ============================================================================
# Document Processing Configuration
# ============================================================================

# Supported file types
SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.docx', '.json', '.md']

# Documents directory
DOCUMENTS_DIR = "./documents"

# ============================================================================
# Agent Configuration
# ============================================================================

# Agent behavior settings
ENABLE_RAG_CLASSIFICATION = True
VERBOSE_MODE = False
SHOW_RETRIEVED_DOCS = False

# ============================================================================
# Paths
# ============================================================================

BASE_DIR = Path(__file__).parent
CHROMA_DB_PATH = BASE_DIR / CHROMA_DB_PATH
DOCUMENTS_DIR = BASE_DIR / DOCUMENTS_DIR

# Create necessary directories
CHROMA_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Prompts
# ============================================================================

CLASSIFICATION_PROMPT = """
You are a strict routing agent.

Use "rag" if:
- The question asks about policies, rules, guidelines, definitions, procedures, or requirements
- The question refers to documents, manuals, guides, contracts, leases, or company information
- The question could reasonably depend on provided documents
- The question is NOT basic math or common knowledge

Use "direct" ONLY if:
- The question is basic math
- The question is general common knowledge (e.g., 2+2, capital of France)
- The question is casual conversation

Question:
{question}

Answer with ONLY one word:
rag or direct
"""

RAG_ANSWER_PROMPT = """You are a helpful assistant. Answer the user's question based ONLY on the provided context.

Context from documents:
{context}

User Question: {query}

Instructions:
- Use ONLY information from the context above
- If the context doesn't contain relevant information, clearly state that
- Be concise and accurate
- Cite specific parts of the context when relevant
- If the answer requires information not in the context, say so

Answer:"""

DIRECT_ANSWER_PROMPT = """You are a helpful assistant. Answer the user's question naturally.

User Question: {query}

Provide a clear, concise, and helpful answer."""