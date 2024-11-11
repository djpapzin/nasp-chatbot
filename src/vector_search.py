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
        """Load the pre-built vector store or create a new one"""
        try:
            # Check if index files exist
            index_file = _self.index_path / "default_index.faiss"
            pkl_file = _self.index_path / "default_index.pkl"
            
            if not (index_file.exists() and pkl_file.exists()):
                st.warning("FAISS index files not found. Creating new vector store...")
                # Initialize empty vector store
                vector_store = FAISS.from_texts(
                    ["Initial document"], 
                    _self.embeddings
                )
                # Save it
                _self.index_path.mkdir(exist_ok=True)
                vector_store.save_local(
                    folder_path=str(_self.index_path),
                    index_name="default_index"
                )
                return vector_store
                
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
