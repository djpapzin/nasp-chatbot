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
    # Add a title to the sidebar
    st.sidebar.markdown('<h2 style="color: white;">Upload Documents</h2>', 
                       unsafe_allow_html=True)
    # Return the file uploader widget
    return st.sidebar.file_uploader(
        "Choose PDF, DOCX, or TXT files", 
        type=["pdf", "docx", "doc", "txt"],  # Allowed file types
        accept_multiple_files=True,  # Allow multiple file uploads
        help="Maximum file size: 200MB"  # Help text
    )

def load_css():
    """Load custom CSS styles for better UI appearance"""
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
    # Initialize chat session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
            st.markdown(f"**{'You' if message['role'] == 'user' else 'NASP Bot'}:** {message['content']}")

    # Chat input
    if prompt := st.chat_input("Ask about your documents"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"**You:** {prompt}")

        # Generate and display bot response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                docs = vector_search.get_relevant_documents(vector_store, prompt)
                if docs:
                    bot_response = llm_handler.generate_response(prompt, docs)
                    if bot_response:
                        st.markdown(f"**NASP Bot:** {bot_response}")
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": bot_response
                        })

def load_and_process_docs(uploaded_files):
    """Loads and processes documents from uploaded files."""
    if uploaded_files:
        with st.spinner('📁 Processing your documents...'):
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_docs = []
            total_files = len(uploaded_files)
            
            for idx, uploaded_file in enumerate(uploaded_files):
                try:
                    # Update status
                    status_text.text(f"Processing file {idx + 1} of {total_files}: {uploaded_file.name}")
                    
                    # Store filename in metadata
                    metadata = {
                        "filename": uploaded_file.name,
                        "source": uploaded_file.name
                    }
                    
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_file.write(uploaded_file.read())
                        temp_file_path = temp_file.name

                    # Load document with metadata
                    if uploaded_file.name.endswith('.pdf'):
                        with st.spinner(f'📄 Reading PDF: {uploaded_file.name}...'):
                            loader = PyPDFLoader(temp_file_path)
                            docs = loader.load()
                            # Add page numbers to metadata
                            for i, doc in enumerate(docs):
                                doc.metadata.update({
                                    **metadata,
                                    "page": i + 1,
                                    "total_pages": len(docs)
                                })
                    elif uploaded_file.name.endswith(('.docx', '.doc')):
                        with st.spinner(f'📄 Reading DOCX: {uploaded_file.name}...'):
                            loader = Docx2txtLoader(temp_file_path)
                            docs = loader.load()
                            for doc in docs:
                                doc.metadata.update(metadata)
                    elif uploaded_file.name.endswith('.txt'):
                        with st.spinner(f'📄 Reading TXT: {uploaded_file.name}...'):
                            loader = TextLoader(temp_file_path)
                            docs = loader.load()
                            for doc in docs:
                                doc.metadata.update(metadata)
                    
                    all_docs.extend(docs)
                    os.remove(temp_file_path)
                    
                    # Update progress bar
                    progress = (idx + 1) / total_files
                    progress_bar.progress(progress)
                    st.success(f"✅ Successfully loaded {len(docs)} pages from {uploaded_file.name}")

                except Exception as e:
                    st.error(f"Error loading {uploaded_file.name}: {e}")

            # Clear status text after completion
            status_text.empty()
            progress_bar.empty()

            if all_docs:
                with st.spinner('🔄 Processing text chunks...'):
                    print("Splitting documents into chunks...")
                    max_context_length = 8192
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_context_length, chunk_overlap=0)
                    splits = text_splitter.split_documents(all_docs)

                    if not splits:
                        st.error("No text chunks were created. Please check the text splitter settings.")
                        return None, None

                    with st.spinner('🧠 Generating embeddings...'):
                        embeddings_model = TogetherEmbeddings()
                        documents = []
                        for split in splits:
                            doc = Document(
                                page_content=split.page_content,
                                metadata=split.metadata
                            )
                            documents.append(doc)

                        embeddings = embeddings_model.embed_documents([doc.page_content for doc in documents])
                        
                        uuids = [str(uuid4()) for _ in documents]
                        docstore = InMemoryDocstore({uuid: doc for uuid, doc in zip(uuids, documents)})
                        index = faiss.IndexFlatL2(len(embeddings[0]))
                        vector_store = FAISS(
                            embedding_function=embeddings_model,
                            index=index,
                            docstore=docstore,
                            index_to_docstore_id={i: uuid for i, uuid in enumerate(uuids)}
                        )

                        vector_store.add_embeddings(
                            text_embeddings=list(zip(uuids, embeddings)), 
                            metadatas=[doc.metadata for doc in documents]
                        )
                
                st.success('✅ All documents processed successfully!')
                return vector_store, embeddings_model
            else:
                st.error("No documents were loaded. Please check the uploaded files.")
                return None, None
    else:
        st.warning("No files uploaded yet.")
        return None, None
