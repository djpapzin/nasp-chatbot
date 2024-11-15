import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from src.ui import UI
from src.llm import LLMHandler
from src.config import CHATBOT_CONFIG, EMBEDDINGS_CONFIG, LANGUAGE_CONFIG, PROMPT_CONFIG
from src.document_manager import WebDocumentManager

# Load environment variables
load_dotenv()

# Initialize vector store and LLM handler
vector_store = None
llm_handler = None

def init_components():
    """Initialize vector store and LLM handler."""
    try:
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            chunk_size=EMBEDDINGS_CONFIG['chunk_size']
        )
        
        # Load FAISS index with allow_dangerous_deserialization
        vector_store = FAISS.load_local(
            folder_path=EMBEDDINGS_CONFIG['persist_directory'],
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        
        llm_handler = LLMHandler()
        return vector_store, llm_handler
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None, None

def create_qa_prompt(language='en'):
    """Create QA prompt based on selected language."""
    current_prompt = PROMPT_CONFIG[language]['system_prompt']
    return ChatPromptTemplate.from_messages([
        ("system", current_prompt),
        ("human", "{input}")
    ])

def main():
    UI.setup_page()
    global vector_store, llm_handler
    vector_store, llm_handler = init_components()
    
    # Get the selected language or use default
    selected_language = st.session_state.get('language', LANGUAGE_CONFIG['default_language'])
    
    # Create QA prompt
    qa_prompt = create_qa_prompt(selected_language)
    
    UI.show_header()
    UI.show_chat_interface(vector_store, llm_handler, qa_prompt)
    UI.show_file_uploader(WebDocumentManager(), vector_store)

if __name__ == "__main__":
    main()