"""
Document Loading Diagnostic
Check what files exist and if they can be loaded
"""

import os
from pathlib import Path
from config import DOCUMENTS_DIR

print("=" * 70)
print("DOCUMENT LOADING DIAGNOSTIC")
print("=" * 70)

# Check if documents directory exists
print(f"\n1. Checking documents directory: {DOCUMENTS_DIR}")
if not Path(DOCUMENTS_DIR).exists():
    print(f"❌ Directory does not exist!")
    exit(1)
else:
    print(f"✓ Directory exists")

# List all files
print(f"\n2. Files in directory:")
files = list(Path(DOCUMENTS_DIR).rglob("*"))
files = [f for f in files if f.is_file()]

if not files:
    print("❌ No files found!")
    exit(1)

for f in files:
    size_kb = f.stat().st_size / 1024
    print(f"  - {f.name} ({f.suffix}) [{size_kb:.1f} KB]")

print(f"\n  Total: {len(files)} files")

# Check dependencies
print(f"\n3. Checking dependencies:")

try:
    from pypdf import PdfReader
    print("  ✓ pypdf installed (PDF support)")
except ImportError:
    print("  ❌ pypdf NOT installed (PDF files will fail)")
    print("     Install with: pip install pypdf")

try:
    from docx import Document
    print("  ✓ python-docx installed (DOCX support)")
except ImportError:
    print("  ⚠ python-docx NOT installed (DOCX files will fail)")

# Try loading with document_loader directly
print(f"\n4. Testing direct document loading (not via MCP):")

try:
    from document_loader import DocumentLoader
    loader = DocumentLoader()
    
    print(f"  Supported extensions: {loader.supported_extensions}")
    
    # Try loading
    print(f"\n  Loading documents...")
    documents = loader.load_directory(str(DOCUMENTS_DIR), recursive=True)
    
    print(f"\n  Result type: {type(documents)}")
    print(f"  Number of documents: {len(documents) if isinstance(documents, list) else 'N/A'}")
    
    if isinstance(documents, list):
        for i, doc in enumerate(documents, 1):
            print(f"\n  Document {i}:")
            print(f"    ID: {doc.get('id', 'N/A')}")
            print(f"    Type: {doc.get('metadata', {}).get('type', 'N/A')}")
            print(f"    Filename: {doc.get('metadata', {}).get('filename', 'N/A')}")
            print(f"    Text length: {len(doc.get('text', ''))}")
    else:
        print(f"  Documents: {documents}")
        
except Exception as e:
    print(f"  ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Try loading via MCP
print(f"\n5. Testing MCP document loading:")

try:
    from mcp_client import DocumentLoaderMCP
    
    loader_mcp = DocumentLoaderMCP()
    documents = loader_mcp.load_directory(str(DOCUMENTS_DIR), recursive=True)
    
    print(f"  Result type: {type(documents)}")
    
    if isinstance(documents, list):
        print(f"  Number of documents: {len(documents)}")
        if documents:
            print(f"\n  First document:")
            print(f"    ID: {documents[0].get('id', 'N/A')}")
            print(f"    Keys: {list(documents[0].keys())}")
    elif isinstance(documents, dict):
        print(f"  ⚠ Got dict instead of list!")
        print(f"  Keys: {list(documents.keys())}")
        if 'error' in documents:
            print(f"  Error: {documents['error']}")
    else:
        print(f"  Documents: {documents}")
        
except Exception as e:
    print(f"  ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)