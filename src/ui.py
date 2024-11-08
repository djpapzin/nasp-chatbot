import streamlit as st
from together import Together
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_together import TogetherEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
import tempfile
from src.llm import LLMHandler

def setup_page():
    """Configure initial page settings for the Streamlit app"""
    # Set basic page configuration including title, icon, and layout
    st.set_page_config(
        page_title="NASP Chatbot",
        page_icon="🤖",
        layout="wide",  # Use wide layout for better space utilization
        initial_sidebar_state="expanded"  # Start with sidebar expanded
    )

def show_header():
    """Display the main header and welcome message at the top of the page"""
    # Display main title with custom styling
    st.markdown('<div class="title-container"><h1 class="title-text">NASP Chatbot</h1></div>', 
                unsafe_allow_html=True)
    
    # Display welcome message with custom styling
    st.markdown("""
        <div class="welcome-box">
            👋 Welcome! This is a prototype chatbot for the National Agency of Social Protection. 
            You can use it to ask questions about a library of reports, evaluations, research and other documents.
        </div>
    """, unsafe_allow_html=True)

def show_how_to_use():
    """Display the How to Use section in an expandable container"""
    with st.expander("How to use"):
        st.markdown("""
            <ol>
                <li>⬅️ Use the sidebar to upload your documents (PDF, DOCX, or TXT)</li>
                <li>Wait for the documents to be processed</li>
                <li>Once processing is complete, the chat interface will appear</li>
                <li>Ask questions about your documents!</li>
            </ol>
        """, unsafe_allow_html=True)

def setup_file_uploader():
    """Setup the file uploader widget in the sidebar"""
    print("\n=== Setting up File Uploader ===")
    
    # Add a title to the sidebar
    st.sidebar.markdown('<h2 style="color: white;">Upload Documents</h2>', 
                       unsafe_allow_html=True)
    
    # Return the file uploader widget
    uploaded_files = st.sidebar.file_uploader(
        "Choose PDF, DOCX, or TXT files", 
        type=["pdf", "docx", "doc", "txt"],
        accept_multiple_files=True,
        help="Maximum file size: 200MB"
    )
    
    if uploaded_files:
        print(f"Files uploaded: {[f.name for f in uploaded_files]}")
    else:
        print("No files uploaded yet")
    
    return uploaded_files

def load_css():
    """Load custom CSS styles"""
    st.markdown("""
        <style>
        /* Main container styling */
        .main {
            padding: 2rem;
        }
        
        /* Title container styling */
        .title-container {
            background-color: #1E3D59;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        /* Title text styling */
        .title-text {
            color: white;
            font-size: 3rem;
            font-weight: bold;
            margin: 0;
        }
        
        /* Welcome box styling */
        .welcome-box {
            background-color: #17A2B8;
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1.5rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* How to use section styling */
        .stExpander {
            margin: 1.5rem 0;
            border: none;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-radius: 10px;
        }
        
        /* Success/Error message styling */
        .stSuccess, .stError {
            padding: 0.5rem !important;
            border-radius: 0.5rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

def show_chat_interface(vector_store, vector_search, llm_handler):
    """Display chat interface and handle messages"""
    print("\n=== Chat Interface Initialization ===")
    
    # Initialize chat session state
    if "messages" not in st.session_state:
        print("Initializing new chat session")
        st.session_state.messages = []
        # Add initial bot message
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello. Please enter your question in the chat box to get started."
        })
        print("Added welcome message to chat")

    # Display chat history
    print(f"\nCurrent chat history: {len(st.session_state.messages)} messages")
    for message in st.session_state.messages:
        print(f"Displaying message from {message['role']}")
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
            st.markdown(f"**{'You' if message['role'] == 'user' else 'NASP Bot'}:** {message['content']}")

    # Chat input
    if prompt := st.chat_input("Ask about your documents"):
        print(f"\n=== New User Input ===\nUser query: {prompt}")
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"**You:** {prompt}")

        # Generate and display bot response
        print("\nGenerating bot response...")
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                docs = vector_search.get_relevant_documents(vector_store, prompt)
                if docs:
                    print(f"Found {len(docs)} relevant documents")
                    bot_response = llm_handler.generate_response(prompt, docs)
                    if bot_response:
                        print("Bot response generated successfully")
                        st.markdown(f"**NASP Bot:** {bot_response}")
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": bot_response
                        })
                else:
                    print("No relevant documents found")

def load_and_process_docs(uploaded_files):
    """Loads and processes documents from uploaded files."""
    print("\n=== Document Processing Started ===")
    
    if uploaded_files:
        print(f"Processing {len(uploaded_files)} files")
        with st.spinner('📁 Processing your documents...'):
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_docs = []
            total_files = len(uploaded_files)
            
            for idx, uploaded_file in enumerate(uploaded_files):
                try:
                    print(f"\nProcessing file {idx + 1}/{total_files}: {uploaded_file.name}")
                    status_text.text(f"Processing file {idx + 1} of {total_files}: {uploaded_file.name}")
                    
                    # Process file based on type
                    if uploaded_file.name.endswith('.pdf'):
                        print("Processing PDF file...")
                    elif uploaded_file.name.endswith(('.docx', '.doc')):
                        print("Processing DOCX file...")
                    elif uploaded_file.name.endswith('.txt'):
                        print("Processing TXT file...")
                    
                    # Update progress
                    progress = (idx + 1) / total_files
                    progress_bar.progress(progress)
                    print(f"Progress: {progress * 100}%")

                except Exception as e:
                    print(f"Error processing file: {str(e)}")
                    st.error(f"Error loading {uploaded_file.name}: {e}")

            print("\n=== Document Processing Complete ===")
            print(f"Total documents processed: {len(all_docs)}")
            
            return all_docs
    else:
        print("No files to process")
        return None
