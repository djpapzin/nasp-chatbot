# Import required libraries
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import re
from typing import List, Iterator
import logging
import time
from datetime import datetime
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from ratelimit import limits, sleep_and_retry
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from src.vector_search import VectorSearch

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Timer:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        print(f"\n⏱️ Starting {self.name}...")
        return self
        
    def __exit__(self, *args):
        elapsed_time = time.time() - self.start_time
        print(f"✓ Completed {self.name} in {elapsed_time:.2f} seconds")

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
        doc_id = 0  # Track unique document IDs
        
        # Group documents by source file
        docs_by_source = {}
        for doc in documents:
            source = doc.metadata.get('source', '')
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append(doc)
        
        # Process each source file's documents
        for source, docs in docs_by_source.items():
            doc_id += 1
            file_name = Path(source).name
            
            # Extract document metadata
            doc_metadata = {
                "document_id": doc_id,
                "title": self._clean_filename(file_name),
                "source": source,
                "file_name": file_name,
                "total_pages": len(docs),
                "creation_date": datetime.fromtimestamp(Path(source).stat().st_mtime).strftime("%Y-%m-%d"),
            }
            
            # Process each page in the document
            all_text = []
            for page_num, doc in enumerate(docs, 1):
                # Clean the text
                clean_content = self._preprocess_text(doc.page_content)
                all_text.append(clean_content)
                
            # Split into chunks
            splits = self.text_splitter.create_documents(
                texts=all_text,
                metadatas=[{
                    **doc_metadata,
                    "page_number": page_num,
                    "total_pages": len(docs)
                } for page_num in range(1, len(docs) + 1)]
            )
            
            # Process each chunk
            for chunk_idx, split in enumerate(splits):
                chunk_metadata = {
                    **split.metadata,
                    "chunk_index": chunk_idx,
                    "chunk_size": len(split.page_content),
                    "total_chunks": len(splits),
                }
                
                processed_docs.append(
                    Document(
                        page_content=split.page_content,
                        metadata=chunk_metadata
                    )
                )
            
            logger.info(f"Processed document {doc_id}: {file_name} - {len(splits)} chunks created")
        
        return processed_docs
    
    def _clean_filename(self, filename: str) -> str:
        """Convert filename to readable title"""
        # Remove extension
        title = Path(filename).stem
        # Replace underscores and hyphens with spaces
        title = title.replace('_', ' ').replace('-', ' ')
        return title
    
    def _preprocess_text(self, text: str) -> str:
        """Clean text while preserving semantic structure"""
        # Remove redundant whitespace
        text = re.sub(r'\s+', ' ', text)
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        return text.strip()

def batch_documents(documents: List[Document], batch_size: int = 50) -> Iterator[List[Document]]:
    """Split documents into batches"""
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception)
)
def create_vector_store_with_retry(documents: List[Document], embeddings) -> FAISS:
    """Create vector store with retry logic"""
    vector_store = None
    
    # Process in batches
    for i, doc_batch in enumerate(batch_documents(documents)):
        print(f"Processing batch {i+1}...")
        
        if vector_store is None:
            vector_store = FAISS.from_documents(
                documents=doc_batch,
                embedding=embeddings,
                normalize_L2=True
            )
        else:
            # Add documents to existing store
            vector_store.add_documents(doc_batch)
            
        # Sleep between batches to respect rate limits
        time.sleep(1)  # Adjust this value based on your rate limits
    
    return vector_store

def main():
    print(f"=== Starting Vector Store Setup at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    with Timer("Total Processing"):
        try:
            # Load environment variables
            load_dotenv()
            
            with Timer("Embedding Model Initialization"):
                embeddings = AzureOpenAIEmbeddings(
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    chunk_size=16  # Process fewer embeddings at once
                )
            
            # Process PDFs
            docs_path = Path("src/default_docs")
            documents = []
            
            with Timer("PDF Loading"):
                for pdf_file in docs_path.glob("*.pdf"):
                    print(f"Processing: {pdf_file.name}")
                    loader = PyPDFLoader(str(pdf_file))
                    documents.extend(loader.load())
            
            if not documents:
                print("No documents found!")
                return
                
            # Process documents
            with Timer("Document Chunking"):
                processor = DocumentProcessor()
                processed_docs = processor.process_documents(documents)
                print(f"Created {len(processed_docs)} chunks")
            
            # Create vector store with batching and retry logic
            with Timer("Vector Store Creation & Embedding"):
                vector_search = VectorSearch()
                vector_search.initialize_embeddings()

                vector_store = create_vector_store_with_retry(processed_docs, vector_search.embeddings)
            
            # Save vector store
            with Timer("Vector Store Saving"):
                vector_store.save_local(
                    folder_path="faiss_index",
                    index_name="index"
                )
            
            print("\n=== Processing Summary ===")
            print(f"• Documents processed: {len(documents)}")
            print(f"• Chunks created: {len(processed_docs)}")
            print(f"• Average chunk size: {sum(len(d.page_content) for d in processed_docs) / len(processed_docs):.0f} characters")
            
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()