import os
from pathlib import Path
from typing import List, Optional, Tuple
import logging
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorSearch:
    def __init__(self):
        self.vector_store = None
        self.embeddings = None
        
    def initialize_embeddings(self):
        """Initialize embeddings model"""
        try:
            self.embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"), 
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY")
            )
            return True
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            return False

    def load_or_create_vector_store(self) -> Optional[FAISS]:
        """Load existing vector store or create new one"""
        try:
            if not self.embeddings:
                self.initialize_embeddings()
                
            # Load existing vector store with safe deserialization enabled
            self.vector_store = FAISS.load_local(
                folder_path="faiss_index",
                embeddings=self.embeddings,
                index_name="index",
                allow_dangerous_deserialization=True
            )
            logger.info("Vector store loaded successfully")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return None

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Perform similarity search"""
        try:
            if not self.vector_store:
                self.vector_store = self.load_or_create_vector_store()
                if not self.vector_store:
                    return []

            # Use FAISS similarity search
            docs = self.vector_store.similarity_search(
                query=query,
                k=k
            )
            
            # Filter and process results
            filtered_docs = []
            seen_content = set()
            
            for doc in docs:
                # Skip if content is too short or is metadata
                if len(doc.page_content.strip()) < 50:
                    continue
                    
                if "Published by" in doc.page_content:
                    continue
                    
                # Skip near-duplicate content
                content_hash = hash(doc.page_content[:200])
                if content_hash in seen_content:
                    continue
                    
                # Add metadata
                doc.metadata["source_file"] = Path(doc.metadata.get("source", "")).name
                filtered_docs.append(doc)
                seen_content.add(content_hash)
                
                if len(filtered_docs) >= k:
                    break
            
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []

    def similarity_search_with_score(self, query: str, k: int = 3) -> List[tuple[Document, float]]:
        """Perform similarity search with scores"""
        try:
            if not self.vector_store:
                self.vector_store = self.load_or_create_vector_store()
                if not self.vector_store:
                    return []

            # Use FAISS similarity search with scores
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []

    @staticmethod
    def initialize() -> Tuple[Optional[FAISS], Optional['VectorSearch']]:
        """Initialize vector store and search components"""
        vector_search = VectorSearch()
        vector_search.initialize_embeddings()
        
        try:
            # Load existing vector store
            vector_store = FAISS.load_local(
                folder_path="faiss_index",
                embeddings=vector_search.embeddings,
                index_name="index",
                allow_dangerous_deserialization=True
            )
            logger.info("Loaded existing vector store")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            vector_store = None
        
        return vector_store, vector_search

    def initialize_vector_store(self):
        try:
            embeddings = self.initialize_embeddings()
            if embeddings is None:
                return None
                
            # Initialize vector store
            vector_store = FAISS.from_documents(
                documents=[],  # Start with empty docs
                embedding=embeddings
            )
            return vector_store
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            return None
