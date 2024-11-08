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
    print("\n=== Document Processing Started ===")
    if not uploaded_files:
        print("No files to process")
        return None
        
    with st.spinner('📁 Processing your documents...'):
        try:
            all_docs = []
            total_files = len(uploaded_files)
            
            for idx, uploaded_file in enumerate(uploaded_files):
                print(f"\nProcessing file {idx + 1}/{total_files}: {uploaded_file.name}")
                
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
                        print("Processing PDF file...")
                        loader = PyPDFLoader(temp_file_path)
                        docs = loader.load()
                        # Add page numbers to metadata
                        for i, doc in enumerate(docs):
                            doc.metadata.update({
                                **metadata,
                                "page": i + 1,
                                "total_pages": len(docs)
                            })
                        all_docs.extend(docs)
                        print(f"Successfully loaded {len(docs)} pages from {uploaded_file.name}")
                        
                except Exception as e:
                    print(f"Error processing file: {str(e)}")
                    st.error(f"Error loading {uploaded_file.name}: {e}")
                    continue
                finally:
                    os.unlink(temp_file_path)

            if not all_docs:
                print("No documents were processed successfully")
                return None

            print("\n=== Splitting Documents ===")
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(all_docs)
            print(f"Created {len(splits)} text chunks")

            return splits

        except Exception as e:
            print(f"Error in document processing: {str(e)}")
            st.error(f"Error processing documents: {str(e)}")
            return None
