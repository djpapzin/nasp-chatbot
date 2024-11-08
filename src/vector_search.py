import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_together import TogetherEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
from uuid import uuid4

class VectorSearch:
    def __init__(self):
        """Initialize vector search with TogetherEmbeddings"""
        self.embeddings_model = TogetherEmbeddings()

    def create_vector_store(self, documents):
        """Create FAISS vector store from documents"""
        try:
            with st.spinner('🧠 Generating embeddings...'):
                # Generate embeddings for all documents
                embeddings = self.embeddings_model.embed_documents(
                    [doc.page_content for doc in documents]
                )
                
                # Generate UUIDs for documents
                uuids = [str(uuid4()) for _ in documents]
                
                # Initialize FAISS vector store
                docstore = InMemoryDocstore({
                    uuid: doc for uuid, doc in zip(uuids, documents)
                })
                index = faiss.IndexFlatL2(len(embeddings[0]))
                
                vector_store = FAISS(
                    embedding_function=self.embeddings_model,
                    index=index,
                    docstore=docstore,
                    index_to_docstore_id={i: uuid for i, uuid in enumerate(uuids)}
                )

                # Add embeddings to vector store
                vector_store.add_embeddings(
                    text_embeddings=list(zip(uuids, embeddings)), 
                    metadatas=[doc.metadata for doc in documents]
                )

                return vector_store, self.embeddings_model

        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return None, None

    def get_relevant_documents(self, vector_store, query, k=3):
        """Retrieve relevant documents for a query"""
        try:
            retriever = vector_store.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": k}
            )
            return retriever.get_relevant_documents(query)
        except Exception as e:
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
