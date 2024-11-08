import os
import requests
from together import Together
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_together import TogetherEmbeddings
from dotenv import load_dotenv
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
import streamlit as st
from streamlit_chat import message
import tempfile

# Load environment variables from .env file
load_dotenv()

def load_and_process_docs(uploaded_files):
    """Loads and processes documents from uploaded files."""
    if uploaded_files:
        all_docs = []
        for uploaded_file in uploaded_files:
            try:
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name

                file_extension = os.path.splitext(uploaded_file.name)[1]
                if file_extension == ".pdf":
                    loader = PyPDFLoader(temp_file_path)
                elif file_extension == ".docx" or file_extension == ".doc":
                    loader = Docx2txtLoader(temp_file_path)
                elif file_extension == ".txt":
                    loader = TextLoader(temp_file_path)
                else:
                    st.error(f"Unsupported file format: {file_extension}")
                    continue

                docs = loader.load()
                all_docs.extend(docs)
                st.success(f"Loaded {len(docs)} pages from {uploaded_file.name}")
                os.remove(temp_file_path)

            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {e}")

        if all_docs:
            max_context_length = 8192
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_context_length, chunk_overlap=0)
            splits = text_splitter.split_documents(all_docs)

            if not splits:
                st.error("No text chunks were created. Please check the text splitter settings.")
                return None, None

            embeddings_model = TogetherEmbeddings()
            
            # Create documents with proper metadata
            documents = []
            for split in splits:
                doc = Document(
                    page_content=split.page_content,
                    metadata=split.metadata  # Use the split's metadata instead
                )
                documents.append(doc)

            # Generate embeddings for all documents
            embeddings = embeddings_model.embed_documents([doc.page_content for doc in documents])
            
            # Generate UUIDs
            uuids = [str(uuid4()) for _ in documents]
            
            # Create docstore
            docstore = InMemoryDocstore({uuid: doc for uuid, doc in zip(uuids, documents)})

            # Create FAISS index
            index = faiss.IndexFlatL2(len(embeddings[0]))
            vector_store = FAISS(
                embedding_function=embeddings_model,
                index=index,
                docstore=docstore,
                index_to_docstore_id={i: uuid for i, uuid in enumerate(uuids)}
            )

            # Add embeddings with matching metadata
            vector_store.add_embeddings(
                text_embeddings=list(zip(uuids, embeddings)), 
                metadatas=[doc.metadata for doc in documents]  # This ensures metadata matches texts
            )
            
            return vector_store, embeddings_model
        else:
            st.error("No documents were loaded. Please check the uploaded files.")
            return None, None
    else:
        st.warning("No files uploaded yet.")
        return None, None



def main():
    load_dotenv()
    api_key = os.getenv("TOGETHER_API_KEY")
    os.environ["TOGETHER_API_KEY"] = api_key
    client = Together()

    st.title("Kwantu Chatbot")

    # Sidebar for document upload
    st.sidebar.title("Upload Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Choose PDF, DOCX, or TXT files", 
        type=["pdf", "docx", "doc", "txt"], 
        accept_multiple_files=True
    )

    # Process documents if uploaded
    vector_store, embeddings_model = load_and_process_docs(uploaded_files)
    
    # Main chat interface
    if vector_store and embeddings_model:  # Only show chat if documents are processed
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        # Initialize chat session state
        if "history" not in st.session_state:
            st.session_state["history"] = []
        if "generated" not in st.session_state:
            st.session_state["generated"] = ["Hello! Ask me anything about your documents 🤗"]
        if "past" not in st.session_state:
            st.session_state["past"] = ["Hey! 👋"]

        # Chat interface
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Documents", key='input')
            submit_button = st.form_submit_button(label='Send')

        reply_container = st.container()
        if submit_button and user_input:
            with st.spinner('Generating response...'):
                context = retriever.get_relevant_documents(user_input)
                formatted_context = "\n\n".join([f"Document {i+1} Content:\n{doc.page_content}" for i, doc in enumerate(context)])

                response = client.chat.completions.create(
                    model="meta-llama/Llama-3.2-3B-Instruct-Turbo",
                    messages=[{"role": "user", "content": f"User Query: {user_input}\n\nContext:\n{formatted_context}"}],
                    max_tokens=512,
                    temperature=0.7,
                    top_p=0.7,
                    top_k=50,
                    repetition_penalty=1,
                    stop=["<|eot_id|>", "<|eom_id|>"],
                    stream=True
                )
                output = ""
                for token in response:
                    if hasattr(token, 'choices'):
                        output += token.choices[0].delta.content

                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state.get('generated'):
            with reply_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                    message(st.session_state['generated'][i], key=str(i), avatar_style="fun-emoji")

    else:
        # Simplified welcome message without extras
        st.info("👋 Welcome! Please upload your documents to start chatting.")
        
        with st.expander("How to use"):
            st.markdown("""
            1. Use the sidebar to upload your documents (PDF, DOCX, or TXT)
            2. Wait for the documents to be processed
            3. Once processing is complete, the chat interface will appear
            4. Ask questions about your documents!
            """)

if __name__ == "__main__":
    main()