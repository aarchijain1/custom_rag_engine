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

# ============================================================================
# Model Configuration
# ============================================================================

# Gemini model settings
GEMINI_MODEL = "gemini-1.5-flash"
TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 2048

# Embedding model (runs locally)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Alternative: "all-mpnet-base-v2" (better quality, slower)

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
ENABLE_RAG_CLASSIFICATION = True  # Let agent decide when to use RAG
VERBOSE_MODE = False  # Show debug information
SHOW_RETRIEVED_DOCS = False  # Display retrieved document chunks

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

CLASSIFICATION_PROMPT = """You are a query classifier. Analyze the user query and decide:

User Query: "{query}"

Should this query be answered using:
A) RAG (document search) - for factual questions about specific topics in documents
B) DIRECT - for general knowledge, greetings, simple questions, or conversational queries

Consider:
- Use RAG for: "What does the document say about X?", "Find information on Y", technical questions
- Use DIRECT for: "Hello", "How are you?", "What is 2+2?", general knowledge questions

Respond with ONLY one word: 'RAG' or 'DIRECT'"""

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