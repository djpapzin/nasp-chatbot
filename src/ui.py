import streamlit as st
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from typing import List
import os
from src.document_manager import WebDocumentManager
from src.config import CHATBOT_CONFIG, PROMPT_CONFIG, LANGUAGE_CONFIG
from src.translator import Translator  # Or the specific function you need

logger = logging.getLogger(__name__)

class UI:
    @staticmethod
    def setup_page():
        """Configure initial page settings"""
        st.set_page_config(
            page_title="NASP Chatbot",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    @staticmethod
    def show_header():
        """Display header and welcome message"""
        st.title("NASP Chatbot")
        st.markdown("""
            Welcome! This is a prototype chatbot for the National Agency of Social Protection. 
            You can use it to ask questions about a library of reports, evaluations, research, and other documents.
        """)

    @staticmethod
    def show_file_uploader(doc_manager, vector_store) -> None:
        """Display file upload interface in sidebar"""
        with st.sidebar:
            st.header("📄 Upload Additional Documents")
            uploaded_files = st.file_uploader(
                "Upload PDF, DOCX, or TXT files",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                help="Maximum file size: 200MB",
                disabled=True  # Disable the uploader
            )
            
            st.warning("Document uploads are not supported yet. Please check back later.")

    @staticmethod
    def format_sources(sources: List[Document]) -> str:
        """Format source documents into a readable string with metadata"""
        formatted_sources = "\n\n### Sources Used:\n"
        for doc in sources:
            formatted_sources += f"- **{doc.metadata['source']}** (Page {doc.metadata.get('page', 'N/A')})\n"
            formatted_sources += f"  > {doc.page_content[:200]}...\n"
        return formatted_sources

    @staticmethod
    def show_chat_interface(vector_store, llm_handler, qa_prompt):
        """Display chat interface with RAG"""
        if vector_store is None:
            st.error("Vector store not initialized. Please check your credentials and try again.")
            return
        
        # Create the RAG chain
        try:
            retriever = vector_store.as_retriever()
            question_answer_chain = create_stuff_documents_chain(
                llm_handler.llm,
                qa_prompt
            )
            qa_chain = create_retrieval_chain(retriever, question_answer_chain)

            # Create container for chat interface
            chat_container = st.container()
            
            with chat_container:
                # Initialize chat history if it doesn't exist
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                    # Add initial greeting with simpler message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Hello. Please enter your question in the chat box to get started."
                    })
                
                # Display all messages in the chat history
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
                
                # Chat input
                if prompt := st.chat_input("What would you like to know?"):
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    
                    # Display user message
                    with st.chat_message("user"):
                        st.write(prompt)
                    
                    # Generate and display assistant response
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            try:
                                response = qa_chain.invoke({
                                    "input": prompt
                                })
                                
                                if response and "answer" in response:
                                    st.write(response["answer"])
                                    # Add assistant response to chat history
                                    st.session_state.messages.append({
                                        "role": "assistant", 
                                        "content": response["answer"]
                                    })
                                    
                                    # Show sources if available
                                    if "context" in response:
                                        with st.expander("📚 View Sources", expanded=False):
                                            formatted_sources = UI.format_sources(response["context"])
                                            st.markdown(formatted_sources)
                                else:
                                    st.error("I couldn't find relevant information to answer your question.")
                                            
                            except Exception as e:
                                st.error("I encountered an error while generating a response. Please try again.")
                                logger.error(f"Error generating response: {str(e)}", exc_info=True)
        except Exception as e:
            st.error(f"Error initializing chat interface: {str(e)}")
            return

def initialize_session_state():
    if 'language' not in st.session_state:
        st.session_state.language = LANGUAGE_CONFIG['default_language']

def main():
    UI.setup_page()
    
    # Language selector at the top of the page, before any other content
    with st.sidebar:
        selected_language = st.selectbox(
            "Choose Language / Выберите язык / Tilni tanlang",
            options=list(LANGUAGE_CONFIG['available_languages'].keys()),
            format_func=lambda x: LANGUAGE_CONFIG['available_languages'][x],
            key='language'
        )
    
    # Initialize translator
    translator = Translator()
    
    # Translate and display welcome message
    welcome_message = CHATBOT_CONFIG['welcome_message']
    translated_welcome = translator.translate(welcome_message, selected_language)
    
    st.title(CHATBOT_CONFIG['name'])
    st.markdown(translated_welcome)
    
    # Create QA prompt based on selected language
    current_prompt = PROMPT_CONFIG[selected_language]['system_prompt']
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", current_prompt),
        ("human", "{input}")
    ])

    # ... initialize other components ...

    UI.show_chat_interface(vector_store, llm_handler, qa_prompt)
