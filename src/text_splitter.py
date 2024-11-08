import tempfile
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_together import TogetherEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
from uuid import uuid4

def process_documents(uploaded_files):
    """Process uploaded documents and create vector store"""
    if not uploaded_files:
        return None, None
        
    with st.spinner('📁 Processing your documents...'):
        try:
            # Create a progress bar
            progress_bar = st.progress(0)

            all_docs = []
            # Process each uploaded file
            for uploaded_file in uploaded_files:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name

                # Store metadata
                metadata = {
                    "filename": uploaded_file.name,
                    "source": uploaded_file.name
                }

                try:
                    # Load document based on file type
                    if uploaded_file.name.endswith('.pdf'):
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
                        loader = Docx2txtLoader(temp_file_path)
                        docs = loader.load()
                        for doc in docs:
                            doc.metadata.update(metadata)
                    elif uploaded_file.name.endswith('.txt'):
                        loader = TextLoader(temp_file_path)
                        docs = loader.load()
                        for doc in docs:
                            doc.metadata.update(metadata)

                    all_docs.extend(docs)
                    os.remove(temp_file_path)
                    st.success(f"✅ Successfully loaded {len(docs)} pages from {uploaded_file.name}")

                except Exception as e:
                    st.error(f"Error loading {uploaded_file.name}: {str(e)}")
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

            if not all_docs:
                st.error("No documents were loaded successfully.")
                return None, None

            # Split documents into chunks
            with st.spinner('🔄 Processing text chunks...'):
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=8192,
                    chunk_overlap=0
                )
                splits = text_splitter.split_documents(all_docs)

            # Generate embeddings
            with st.spinner('🧠 Generating embeddings...'):
                embeddings_model = TogetherEmbeddings()
                
                # Create documents with metadata
                documents = []
                for split in splits:
                    doc = Document(
                        page_content=split.page_content,
                        metadata=split.metadata
                    )
                    documents.append(doc)

                # Generate embeddings and create vector store
                embeddings = embeddings_model.embed_documents([doc.page_content for doc in documents])
                uuids = [str(uuid4()) for _ in documents]
                
                # Initialize FAISS vector store
                docstore = InMemoryDocstore({uuid: doc for uuid, doc in zip(uuids, documents)})
                index = faiss.IndexFlatL2(len(embeddings[0]))
                vector_store = FAISS(
                    embedding_function=embeddings_model,
                    index=index,
                    docstore=docstore,
                    index_to_docstore_id={i: uuid for i, uuid in enumerate(uuids)}
                )

                # Add embeddings to vector store
                vector_store.add_embeddings(
                    text_embeddings=list(zip(uuids, embeddings)), 
                    metadatas=[doc.metadata for doc in documents]
                )

            st.success('✅ All documents processed successfully!')
            return vector_store, embeddings_model

        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            return None, None
