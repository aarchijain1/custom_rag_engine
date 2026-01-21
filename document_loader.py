"""
Document Loader - Load documents from various file formats
Supports: TXT, PDF, DOCX, JSON, Markdown
"""

from pathlib import Path
from typing import List, Dict, Optional
import json
from config import SUPPORTED_EXTENSIONS

# Optional imports with graceful fallback
try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


class DocumentLoader:
    """Load and parse documents from various formats"""
    
    def __init__(self):
        self.supported_extensions = SUPPORTED_EXTENSIONS
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check which optional dependencies are available"""
        if not PDF_AVAILABLE:
            print("⚠ Warning: pypdf not installed. PDF support disabled.")
            print("  Install with: pip install pypdf")
        
        if not DOCX_AVAILABLE:
            print("⚠ Warning: python-docx not installed. DOCX support disabled.")
            print("  Install with: pip install python-docx")
    
    def load_txt(self, file_path: Path) -> Dict:
        """
        Load plain text file
        
        Args:
            file_path: Path to text file
            
        Returns:
            Document dictionary
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                "id": file_path.stem,
                "text": content,
                "metadata": {
                    "source": str(file_path),
                    "type": "txt",
                    "filename": file_path.name
                }
            }
        except Exception as e:
            raise Exception(f"Error loading TXT file: {str(e)}")
    
    def load_markdown(self, file_path: Path) -> Dict:
        """
        Load Markdown file
        
        Args:
            file_path: Path to markdown file
            
        Returns:
            Document dictionary
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                "id": file_path.stem,
                "text": content,
                "metadata": {
                    "source": str(file_path),
                    "type": "markdown",
                    "filename": file_path.name
                }
            }
        except Exception as e:
            raise Exception(f"Error loading Markdown file: {str(e)}")
    
    def load_pdf(self, file_path: Path) -> Dict:
        """
        Load PDF file
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Document dictionary
        """
        if not PDF_AVAILABLE:
            raise ImportError("pypdf not installed. Run: pip install pypdf")
        
        try:
            reader = PdfReader(file_path)
            text_parts = []
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            full_text = "\n\n".join(text_parts)
            
            return {
                "id": file_path.stem,
                "text": full_text,
                "metadata": {
                    "source": str(file_path),
                    "type": "pdf",
                    "filename": file_path.name,
                    "num_pages": len(reader.pages)
                }
            }
        except Exception as e:
            raise Exception(f"Error loading PDF file: {str(e)}")
    
    def load_docx(self, file_path: Path) -> Dict:
        """
        Load DOCX file
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            Document dictionary
        """
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx not installed. Run: pip install python-docx")
        
        try:
            doc = DocxDocument(file_path)
            
            # Extract text from paragraphs
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            full_text = "\n\n".join(paragraphs)
            
            return {
                "id": file_path.stem,
                "text": full_text,
                "metadata": {
                    "source": str(file_path),
                    "type": "docx",
                    "filename": file_path.name,
                    "num_paragraphs": len(paragraphs)
                }
            }
        except Exception as e:
            raise Exception(f"Error loading DOCX file: {str(e)}")
    
    def load_json(self, file_path: Path) -> Dict:
        """
        Load JSON file
        Expected format: {"text": "...", "metadata": {...}} or any JSON
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Document dictionary
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle structured JSON with 'text' field
            if isinstance(data, dict) and "text" in data:
                return {
                    "id": data.get("id", file_path.stem),
                    "text": data["text"],
                    "metadata": {
                        **data.get("metadata", {}),
                        "source": str(file_path),
                        "type": "json",
                        "filename": file_path.name
                    }
                }
            # Handle list of documents
            elif isinstance(data, list):
                # Combine all items into one document
                combined_text = json.dumps(data, indent=2)
                return {
                    "id": file_path.stem,
                    "text": combined_text,
                    "metadata": {
                        "source": str(file_path),
                        "type": "json",
                        "filename": file_path.name,
                        "num_items": len(data)
                    }
                }
            # Handle any other JSON
            else:
                return {
                    "id": file_path.stem,
                    "text": json.dumps(data, indent=2),
                    "metadata": {
                        "source": str(file_path),
                        "type": "json",
                        "filename": file_path.name
                    }
                }
        except Exception as e:
            raise Exception(f"Error loading JSON file: {str(e)}")
    
    def load_file(self, file_path: str) -> Dict:
        """
        Load a single file based on its extension
        
        Args:
            file_path: Path to file
            
        Returns:
            Document dictionary
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")
        
        extension = path.suffix.lower()
        
        # Route to appropriate loader
        if extension == '.txt':
            return self.load_txt(path)
        elif extension == '.md':
            return self.load_markdown(path)
        elif extension == '.pdf':
            return self.load_pdf(path)
        elif extension == '.docx':
            return self.load_docx(path)
        elif extension == '.json':
            return self.load_json(path)
        else:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported: {', '.join(self.supported_extensions)}"
            )
    
    def load_directory(self, directory_path: str, recursive: bool = True) -> List[Dict]:
        """
        Load all supported files from a directory
        
        Args:
            directory_path: Path to directory
            recursive: Whether to search subdirectories
            
        Returns:
            List of document dictionaries
        """
        path = Path(directory_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not path.is_dir():
            raise ValueError(f"Not a directory: {directory_path}")
        
        documents = []
        errors = []
        
        # Choose search method
        search_method = path.rglob if recursive else path.glob
        
        # Find all supported files
        for file_path in search_method("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    doc = self.load_file(str(file_path))
                    documents.append(doc)
                    print(f"✓ Loaded: {file_path.name}")
                except Exception as e:
                    error_msg = f"✗ Error loading {file_path.name}: {str(e)}"
                    print(error_msg)
                    errors.append(error_msg)
        
        # Summary
        print(f"\n--- Loading Summary ---")
        print(f"Successfully loaded: {len(documents)} documents")
        if errors:
            print(f"Failed: {len(errors)} files")
        
        return documents
    
    def create_document(self, doc_id: str, text: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Create a document dictionary manually
        
        Args:
            doc_id: Unique document ID
            text: Document text
            metadata: Optional metadata
            
        Returns:
            Document dictionary
        """
        return {
            "id": doc_id,
            "text": text,
            "metadata": metadata or {"type": "manual"}
        }


# ============================================================================
# Utility Functions
# ============================================================================

def create_sample_documents(output_dir: str = "./documents"):
    """Create sample documents for testing"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sample 1: Python programming
    with open(output_path / "python_guide.txt", "w") as f:
        f.write("""Python Programming Guide

Python is a high-level, interpreted programming language created by Guido van Rossum 
and first released in 1991. It emphasizes code readability and allows programmers to 
express concepts in fewer lines of code than languages like C++ or Java.

Key Features:
- Simple and easy to learn syntax
- Extensive standard library
- Supports multiple programming paradigms
- Large community and ecosystem

Python is widely used in web development, data science, artificial intelligence, 
scientific computing, and automation.""")
    
    # Sample 2: Machine Learning
    with open(output_path / "ml_intro.txt", "w") as f:
        f.write("""Introduction to Machine Learning

Machine learning is a branch of artificial intelligence focused on building systems 
that can learn from data. Instead of being explicitly programmed, these systems 
improve their performance through experience.

Types of Machine Learning:
1. Supervised Learning - Learning from labeled data
2. Unsupervised Learning - Finding patterns in unlabeled data
3. Reinforcement Learning - Learning through trial and error

Popular frameworks include TensorFlow, PyTorch, and scikit-learn.""")
    
    # Sample 3: JSON format
    json_data = {
        "id": "ai_history",
        "text": """The History of Artificial Intelligence

AI research began in the 1950s with pioneers like Alan Turing asking 'Can machines think?'
The field has experienced several boom and bust cycles, known as 'AI winters' and 'AI summers'.

Recent advances in deep learning, starting around 2012, have led to breakthroughs in:
- Computer vision
- Natural language processing
- Game playing (Chess, Go)
- Autonomous vehicles

Today, AI is transforming industries from healthcare to finance to entertainment.""",
        "metadata": {
            "author": "AI Researcher",
            "date": "2024",
            "topic": "AI History"
        }
    }
    
    with open(output_path / "ai_history.json", "w") as f:
        json.dump(json_data, f, indent=2)
    
    print(f"✓ Created sample documents in: {output_path}")
    print(f"  - python_guide.txt")
    print(f"  - ml_intro.txt")
    print(f"  - ai_history.json")


# ============================================================================
# Standalone Usage
# ============================================================================

if __name__ == "__main__":
    # Create sample documents
    create_sample_documents()
    
    # Test loading
    print("\n--- Testing Document Loader ---")
    loader = DocumentLoader()
    
    # Load directory
    docs = loader.load_directory("./documents")
    
    print(f"\n--- Loaded Documents ---")
    for doc in docs:
        print(f"\nID: {doc['id']}")
        print(f"Type: {doc['metadata']['type']}")
        print(f"Text preview: {doc['text'][:100]}...")