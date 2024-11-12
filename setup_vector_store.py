# Import required libraries
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_together import TogetherEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import re
from typing import List
import logging

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""],
            is_separator_regex=False
        )
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        processed_docs = []
        
        for doc in documents:
            # Clean the text
            clean_content = self._preprocess_text(doc.page_content)
            
            # Split into chunks
            splits = self.text_splitter.create_documents(
                texts=[clean_content],
                metadatas=[doc.metadata]
            )
            
            # Process each chunk
            for i, split in enumerate(splits):
                metadata = {
                    **split.metadata,
                    "chunk_index": i,
                    "chunk_size": len(split.page_content),
                    "source_file": Path(split.metadata.get("source", "")).name
                }
                
                processed_docs.append(
                    Document(
                        page_content=split.page_content,
                        metadata=metadata
                    )
                )
        
        return processed_docs
    
    def _preprocess_text(self, text: str) -> str:
        """Clean text while preserving semantic structure"""
        # Remove redundant whitespace
        text = re.sub(r'\s+', ' ', text)
        # Normalize line endings
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

def create_vector_store(documents: List[Document], embeddings) -> FAISS:
    """Create optimized vector store"""
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
        normalize_L2=True
    )
    return vector_store

def main():
    print("=== Starting Vector Store Setup ===")
    
    try:
        # Load environment variables
        load_dotenv()
        
        # Initialize embeddings
        embeddings = TogetherEmbeddings(
            model="togethercomputer/m2-bert-80M-8k-retrieval",
            together_api_key=os.getenv("TOGETHER_API_KEY")
        )
        
        # Process PDFs
        docs_path = Path("src/default_docs")
        documents = []
        
        for pdf_file in docs_path.glob("*.pdf"):
            print(f"Processing: {pdf_file.name}")
            loader = PyPDFLoader(str(pdf_file))
            documents.extend(loader.load())
        
        if not documents:
            print("No documents found!")
            return
            
        # Process documents
        processor = DocumentProcessor()
        processed_docs = processor.process_documents(documents)
        print(f"Created {len(processed_docs)} chunks")
        
        # Create vector store
        vector_store = create_vector_store(processed_docs, embeddings)
        
        # Save vector store
        vector_store.save_local(
            folder_path="faiss_index",
            index_name="default_index"
        )
        print("Vector store saved successfully")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()