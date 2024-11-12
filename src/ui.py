import streamlit as st
from typing import List
import os
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.retrieval_qa.chain import RetrievalQAChain
import logging

# Configure logger
logger = logging.getLogger(__name__)

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

def setup_file_uploader(vector_store, doc_manager) -> List:
    """Setup file upload widget with document processing"""
    st.sidebar.markdown("### 📄 Upload Additional Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.sidebar.markdown("### 📚 Processing Documents")
        for uploaded_file in uploaded_files:
            with st.sidebar.status(f"Processing {uploaded_file.name}...", expanded=True) as status:
                try:
                    # Process document directly with uploaded file
                    success, result = doc_manager.process_file(uploaded_file, uploaded_file.name)
                    
                    if success:
                        # Add to vector store and save
                        vector_store.add_documents(result)
                        vector_store.save_local("faiss_index", "default_index")
                        status.update(label=f"✅ {uploaded_file.name} processed successfully!", state="complete")
                    else:
                        status.update(label=f"❌ Error processing {uploaded_file.name}: {result}", state="error")
                        
                except Exception as e:
                    status.update(label=f"❌ Error processing {uploaded_file.name}: {str(e)}", state="error")
                    st.sidebar.error(f"Failed to process {uploaded_file.name}")
    
    return uploaded_files

def show_chat_interface(vector_store, llm_handler):
    """Display chat interface with RAG"""
    st.markdown("### 💬 Chat")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Create retriever with similarity search
    retriever = vector_store.as_retriever(
        search_type="similarity",  # Changed from mmr to similarity
        search_kwargs={
            "k": 4  # Number of documents to retrieve
        }
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant for the National Agency of Social Protection.
        Use the following pieces of context to answer the question. 
        If you don't know the answer, just say that you don't know.
        Keep your answers concise and relevant.
        
        Context: {context}"""),
        ("human", "{input}")  # Changed from {question} to {input}
    ])
    
    # Create document chain first
    document_chain = create_stuff_documents_chain(
        llm=llm_handler.llm,
        prompt=prompt,
        document_variable_name="context"  # Specify the variable name for context
    )
    
    # Create retrieval chain
    qa_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain
    )

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if user_input := st.chat_input("Ask your question here"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get response from RAG chain
                    response = qa_chain.invoke({
                        "input": user_input
                    })
                    
                    # Display response
                    st.markdown(response["answer"])
                    
                    # Show sources if available
                    if "context" in response:
                        with st.expander("View Sources"):
                            for doc in response["context"]:
                                st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                                st.markdown(f"**Content:** {doc.page_content}")
                                st.markdown("---")
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response["answer"]
                    })
                    
                except Exception as e:
                    logger.error(f"Error generating response: {str(e)}", exc_info=True)
                    st.error("I encountered an error while generating a response. Please try again.")

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
