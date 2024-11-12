import os
from dotenv import load_dotenv
import streamlit as st
from src.vector_search import VectorSearch
from src.document_manager import DocumentManager
from src.llm import LLMHandler

def initialize_vector_store():
    """Initialize vector store with default documents"""
    vector_search = VectorSearch()
    vector_store = vector_search.load_or_create_vector_store()
    
    if vector_store is None:
        st.error("""
        Failed to initialize vector store. 
        Please check that the FAISS index files are present in the repository.
        """)
        st.stop()
        
    return vector_store, vector_search

def main():
    # Load environment variables
    load_dotenv()
    os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY")
    
    # Page config
    st.set_page_config(
        page_title="NASP Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize components
    vector_store, vector_search = initialize_vector_store()
    llm_handler = LLMHandler()
    doc_manager = DocumentManager()

    # Sidebar for file uploads
    with st.sidebar:
        st.header("📄 Upload Additional Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, or TXT files",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="Maximum file size: 200MB"
        )
        
        # Process uploaded files
        if uploaded_files:
            for file in uploaded_files:
                with st.spinner(f'Processing {file.name}...'):
                    try:
                        success, result = doc_manager.process_file(file, file.name)
                        if success and vector_store is not None:
                            vector_store.add_documents(result)
                            st.success(f'Successfully processed: {file.name}')
                        else:
                            st.error(f'Error processing {file.name}: {result}')
                    except Exception as e:
                        st.error(f'Error processing {file.name}: {str(e)}')
    
    # Main content
    st.title("NASP Chatbot")
    st.markdown("""
        Welcome! This is a prototype chatbot for the National Agency of Social Protection. 
        You can use it to ask questions about a library of reports, evaluations, research, and other documents.
    """)
    
    # Chat interface
    if vector_store is not None:
        st.markdown("### 💬 Ask a question")
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("What would you like to know?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Get scored documents
                    docs_and_scores = vector_search.similarity_search_with_score(
                        prompt,
                        k=4
                    )
                    # Generate response with scored documents
                    response = llm_handler.generate_response(prompt, docs_and_scores)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()