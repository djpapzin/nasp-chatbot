import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import MultiQueryRetriever
from langchain_core.documents import Document
import re
from typing import List
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,          # Smaller chunks for more precise retrieval
            chunk_overlap=50,        # Minimal overlap to maintain context
            length_function=len,
            separators=[
                "\n\n",             # Paragraphs
                "\n",               # Lines
                ".",                # Sentences
                "?",                # Questions
                "!",               # Exclamations
                ";",               # Semi-colons
                ":",               # Colons
                " ",               # Words
                ""                 # Characters
            ]
        )
    
    def clean_text(self, text: str) -> str:
        """Clean the text content"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()
    
    def should_keep_chunk(self, text: str) -> bool:
        """Determine if a chunk should be kept using general rules"""
        # Skip if too short (less than 50 characters)
        if len(text.strip()) < 50:
            return False
            
        # Skip common document metadata and headers
        skip_patterns = [
            r"^\s*page\s+\d+\s*$",  # Page numbers
            r"^\s*\d+\s*$",         # Standalone numbers
            r"^https?://",          # URLs
            r"@.*\.[a-z]{2,}",      # Email addresses
            r"^\s*telephone|fax|email|website|address|copyright|all rights reserved",
            r"^\s*table of contents",
        ]
        
        for pattern in skip_patterns:
            if re.search(pattern, text.lower()):
                return False
                
        # Ensure chunk has meaningful content (at least 10 words)
        words = len(text.split())
        if words < 10:
            return False
                
        return True
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents into clean, relevant chunks"""
        processed_docs = []
        seen_content = set()  # Track unique content
        
        for doc in documents:
            clean_content = self.clean_text(doc.page_content)
            
            # Skip if we've seen this content before
            content_hash = hash(clean_content)
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)
            
            # Add meaningful metadata
            metadata = {
                **doc.metadata,
                "source_file": Path(doc.metadata.get("source", "")).name,
                "content_length": len(clean_content),
                "word_count": len(clean_content.split())
            }
            
            processed_docs.append(
                Document(
                    page_content=clean_content,
                    metadata=metadata
                )
            )
        
        return processed_docs

def test_retriever():
    """Test the retriever functionality"""
    print("\n=== Testing Vector Store Retrieval ===")
    
    try:
        # Initialize embeddings and document processor
        load_dotenv()
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
        
        # Load vector store
        print("Loading vector store...")
        vector_store = FAISS.load_local(
            folder_path="faiss_index",
            embeddings=embeddings,
            index_name="default_index",
            allow_dangerous_deserialization=True
        )
        logger.info("Vector store loaded successfully")
        
        # Test questions
        questions = [
            # Universal Health Insurance questions
            "What is the best way to finance universal health insurance in Uzbekistan?",
            "What should Uzbekistan do to strengthen its overall social protection system?",
            
            # SPIL Project questions
            "What are the objectives of the Social Protection Innovation and Learning project in Uzbekistan?",
            "Which organisations are involved in this project?"
        ]
        
        # Test retrieval for each question
        for question in questions:
            print(f"\nQuestion: {question}")
            try:
                # Get documents using similarity search with scores
                docs_and_scores = vector_store.similarity_search_with_score(
                    query=question,
                    k=3  # Retrieve top 3 most relevant documents
                )
                
                print(f"Found {len(docs_and_scores)} relevant documents:\n")
                
                # Print each document with its score
                for i, (doc, score) in enumerate(docs_and_scores, 1):
                    print(f"Document {i} (Similarity Score: {score:.4f}):")
                    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
                    print(f"Chunk Index: {doc.metadata.get('chunk_index', 'N/A')}")
                    print(f"Chunk Size: {doc.metadata.get('chunk_size', 'N/A')}")
                    print(f"Content: {doc.page_content[:200]}...")
                    print()
                    
            except Exception as e:
                logger.error(f"Error retrieving documents: {str(e)}", exc_info=True)
                print(f"Error: Failed to retrieve documents")
                
    except Exception as e:
        logger.error(f"Error in test_retriever: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")

def display_results(docs: List[Document]):
    """Format and display retrieved documents"""
    print(f"\nFound {len(docs)} relevant documents:\n")
    
    for i, doc in enumerate(docs, 1):
        print(f"Document {i}:")
        print(f"Source: {doc.metadata.get('source_file', 'Unknown')}")
        print(f"Length: {doc.metadata.get('content_length')} chars")
        print(f"Words: {doc.metadata.get('word_count')} words")
        print("Content:", doc.page_content[:200], "..." if len(doc.page_content) > 200 else "")
        print("-" * 80 + "\n")

if __name__ == "__main__":
    test_retriever()
