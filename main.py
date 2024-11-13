import os
from dotenv import load_dotenv
from src.ui import UI
from src.document_manager import WebDocumentManager  # Changed from DocumentManager
from src.llm import LLMHandler
from src.vector_search import VectorSearch

def main():
    # Load environment variables
    load_dotenv()
    os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY")
    
    # Initialize UI
    UI.setup_page()
    
    # Initialize components
    vector_store, vector_search = VectorSearch.initialize()
    llm_handler = LLMHandler()
    doc_manager = WebDocumentManager()  # Changed from DocumentManager
    
    # Show UI components
    UI.show_header()
    UI.show_file_uploader(doc_manager, vector_store)
    UI.show_chat_interface(vector_store, llm_handler)

if __name__ == "__main__":
    main()