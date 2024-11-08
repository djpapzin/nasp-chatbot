import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_together import TogetherEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
import faiss
from uuid import uuid4

class VectorSearch:
    def __init__(self):
        """Initialize vector search with TogetherEmbeddings"""
        print("\n=== Initializing VectorSearch ===")
        self.embeddings_model = TogetherEmbeddings()

    def create_vector_store(self, documents):
        """Create FAISS vector store from documents"""
        print("\n=== Creating Vector Store ===")
        try:
            with st.spinner('🧠 Generating embeddings...'):
                print(f"Processing {len(documents)} documents")
                
                # Ensure documents are in the correct format
                formatted_docs = []
                for doc in documents:
                    if isinstance(doc, Document):
                        formatted_docs.append(doc)
                    else:
                        # Create Document object if not already one
                        formatted_docs.append(Document(
                            page_content=str(doc),
                            metadata=getattr(doc, 'metadata', {})
                        ))
                
                print(f"Formatted {len(formatted_docs)} documents")
                
                # Generate embeddings and create vector store
                texts = [doc.page_content for doc in formatted_docs]
                metadatas = [doc.metadata for doc in formatted_docs]
                
                vector_store = FAISS.from_texts(
                    texts=texts,
                    embedding=self.embeddings_model,
                    metadatas=metadatas
                )
                
                print("Vector store created successfully")
                return vector_store, self.embeddings_model

        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            st.error(f"Error creating vector store: {str(e)}")
            return None, None

    def get_relevant_documents(self, vector_store, query, k=3):
        """Retrieve relevant documents for a query"""
        print(f"\n=== Retrieving Documents for Query: {query} ===")
        try:
            retriever = vector_store.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": k}
            )
            docs = retriever.get_relevant_documents(query)
            print(f"Found {len(docs)} relevant documents")
            return docs
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            st.error(f"Error retrieving documents: {str(e)}")
            return []

    def create_document_summary(self, docs):
        """Create a summary of available documents"""
        doc_summary = {}
        for doc in docs:
            filename = doc.metadata.get('filename', 'N/A')
            if filename not in doc_summary:
                doc_summary[filename] = {
                    'total_pages': doc.metadata.get('total_pages', 'N/A'),
                    'pages_referenced': set([doc.metadata.get('page', 1)])
                }
            else:
                doc_summary[filename]['pages_referenced'].add(
                    doc.metadata.get('page', 1)
                )

        # Format document summary
        docs_context = "Available documents:\n"
        for filename, info in doc_summary.items():
            docs_context += f"- {filename} (Total pages: {info['total_pages']})\n"
        
        return docs_context
