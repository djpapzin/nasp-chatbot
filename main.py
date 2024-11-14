import os
from dotenv import load_dotenv
import streamlit as st
from src.ui import UI
from src.document_manager import WebDocumentManager
from src.llm import LLMHandler
from src.vector_search import VectorSearch
from src.config import CHATBOT_CONFIG, PROMPT_CONFIG, LANGUAGE_CONFIG
from langchain_core.prompts import ChatPromptTemplate

def initialize_session_state():
    if 'language' not in st.session_state:
        st.session_state.language = LANGUAGE_CONFIG['default_language']

def main():
    initialize_session_state()

    # Load environment variables
    load_dotenv()
    os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY")
    
    # Initialize UI
    UI.setup_page()
    
    # Initialize components
    vector_store, vector_search = VectorSearch.initialize()
    llm_handler = LLMHandler()
    doc_manager = WebDocumentManager()

    # Create QA prompt based on selected language
    current_prompt = PROMPT_CONFIG[st.session_state.language]['system_prompt']
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", current_prompt),
        ("human", "{input}")
    ])

    # Show UI components
    UI.show_header()
    UI.show_file_uploader(doc_manager, vector_store)
    UI.show_chat_interface(vector_store, llm_handler, qa_prompt)

if __name__ == "__main__":
    main()