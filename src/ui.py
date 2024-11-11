import streamlit as st
from typing import List
import os

def setup_page():
    """Configure initial page settings"""
    st.set_page_config(
        page_title="NASP Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def show_header():
    """Display header and welcome message"""
    st.markdown('<div class="title-container"><h1 class="title-text">NASP Chatbot</h1></div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
        <div class="welcome-box">
            👋 Welcome! This is a prototype chatbot for the National Agency of Social Protection. 
            You can use it to ask questions about a library of reports, evaluations, research, and other documents.
        </div>
    """, unsafe_allow_html=True)

def show_how_to_use():
    """Display usage instructions"""
    with st.expander("ℹ️ How to use", expanded=False):
        st.markdown("""
        **How to Use the NASP Chatbot:**

        1. **Pre-loaded Documents**: The chatbot comes pre-loaded with key social protection documents.
        2. **Upload Additional Documents**: You can upload additional documents in PDF, DOCX, or TXT format.
        3. **Ask Questions**: Type your questions naturally about any loaded document.
        4. **Receive Answers**: The bot will provide answers based on the document content.

        **Pre-loaded Documents:**
        - *[Exploring Pathways to Decent Employment, Formality and Inclusion in Central Asia](https://socialprotection.org/discover/publications/exploring-pathways-decent-employment-formality-and-inclusion-central-asia)*
        - *[The “Social Protection Innovation and Learning” project in Uzbekistan](https://socialprotection.org/discover/publications/%E2%80%9Csocial-protection-innovation-and-learning%E2%80%9D-project-uzbekistan)*
        - *[Uzbekistan Public Expenditure Review: Better Value for Money in Human Capital and Water Infrastructure](https://socialprotection.org/discover/publications/uzbekistan-public-expenditure-review-better-value-money-human-capital-and)*
        - *[Valuing and investing in unpaid care and domestic work - country case study: Uzbekistan](https://socialprotection.org/discover/publications/valuing-and-investing-unpaid-care-and-domestic-work-country-case-study)*
        - *[Prioritising universal health insurance in Uzbekistan (One pager)](https://socialprotection.org/discover/publications/prioritising-universal-health-insurance-uzbekistan-one-pager)*
        """)

def setup_file_uploader() -> List:
    """Setup file upload widget"""
    st.sidebar.markdown("### 📄 Upload Additional Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.sidebar.markdown("### 📚 Uploaded Documents")
        for file in uploaded_files:
            st.sidebar.markdown(f"- {file.name}")
    
    return uploaded_files

def show_chat_interface(vector_store, vector_search, llm_handler):
    """Display chat interface"""
    st.markdown("### 💬 Chat")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask your question here"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get relevant documents
        docs = vector_search.similarity_search(vector_store, prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = llm_handler.generate_response(prompt, docs)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

def load_css():
    """Load custom CSS"""
    st.markdown("""
        <style>
        .title-container {
            text-align: center;
            padding: 1rem;
        }
        .title-text {
            color: #2e7d32;
        }
        .welcome-box {
            padding: 1rem;
            background-color: #f5f5f5;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
