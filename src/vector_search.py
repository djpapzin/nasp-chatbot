import os
from pathlib import Path
import streamlit as st
from langchain_together import TogetherEmbeddings
from langchain_community.vectorstores import FAISS

class VectorSearch:
    def __init__(self):
        self.embeddings = TogetherEmbeddings(
            model="togethercomputer/m2-bert-80M-8k-retrieval",
            together_api_key=os.getenv("TOGETHER_API_KEY")
        )
        self.index_path = Path("faiss_index")
        
    @st.cache_resource
    def load_or_create_vector_store(_self):
        """Load the pre-built vector store"""
        try:
            return FAISS.load_local(
                folder_path=str(_self.index_path),
                embeddings=_self.embeddings,
                index_name="index",
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            st.error("Please ensure the FAISS index has been properly initialized.")
            return None

    def similarity_search_with_score(self, vector_store, query, k=4):
        """Enhanced similarity search with scores"""
        try:
            docs_and_scores = vector_store.similarity_search_with_score(
                query,
                k=k,
                fetch_k=20
            )
            return docs_and_scores
        except Exception as e:
            print(f"Error in similarity search: {str(e)}")
            return []
