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
                index_name="default_index",
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            st.error("Please ensure the FAISS index has been properly initialized.")
            return None
