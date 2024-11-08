import os
from dotenv import load_dotenv
import streamlit as st
from together import Together
from src.ui import (
    setup_page, 
    show_header, 
    show_how_to_use, 
    setup_file_uploader, 
    load_css,
    show_chat_interface
)
from src.text_splitter import process_documents
from src.vector_search import VectorSearch
from src.llm import LLMHandler

def main():
    # Load environment variables and initialize clients
    load_dotenv()
    api_key = os.getenv("TOGETHER_API_KEY")
    os.environ["TOGETHER_API_KEY"] = api_key
    
    # Initialize components
    vector_search = VectorSearch()
    llm_handler = LLMHandler()
    
    # Setup UI components
    setup_page()
    load_css()
    show_header()
    show_how_to_use()
    
    # Setup file upload
    uploaded_files = setup_file_uploader()
    
    # Process documents if uploaded
    if uploaded_files:
        documents = process_documents(uploaded_files)
        if documents:
            # Create vector store and show chat interface
            vector_store, embeddings_model = vector_search.create_vector_store(documents)
            if vector_store:
                show_chat_interface(vector_store, vector_search, llm_handler)

if __name__ == "__main__":
    main()