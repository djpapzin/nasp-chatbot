from pathlib import Path

# Define the root directory of the project
ROOT_DIR = Path(__file__).parent.parent

# Define paths to important directories
DEFAULT_DOCS_DIR = ROOT_DIR / "src" / "default_docs"
FAISS_INDEX_DIR = ROOT_DIR / "faiss_index"

# Make sure these directories exist
DEFAULT_DOCS_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
