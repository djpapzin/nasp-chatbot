# Import required libraries
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_together import TogetherEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
from ratelimit import limits, sleep_and_retry

# Define rate limit: 10 calls per second
@sleep_and_retry
@limits(calls=10, period=1)
def rate_limited_embedding(embeddings, text):
    """Rate-limited embedding creation"""
    return embeddings.embed_query(text)

def preprocess_text(text):
    """Clean and prepare text for embedding"""
    # Remove extra whitespace
    text = " ".join(text.split())
    # Remove special characters but keep sentence structure
    text = re.sub(r'[^\w\s\.\,\?\!]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_documents_in_batches(documents, batch_size=10):
    """Process documents in batches for faster embedding"""
    for i in range(0, len(documents), batch_size):
        # Get a batch of documents
        batch = documents[i:i + batch_size]
        # Process batch and show progress
        print(f"Processing batch {i//batch_size + 1}/{len(documents)//batch_size + 1}")
        yield batch

def embed_documents(documents):
    """Embed multiple documents in parallel using thread pool"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Process multiple documents simultaneously
        futures = [executor.submit(process_document, doc) for doc in documents]
        # Collect results from all threads
        results = [f.result() for f in futures]
    return results

def create_vector_store(split_docs, embeddings):
    """Create vector store without batch processing"""
    total_chunks = len(split_docs)
    
    print(f"\n=== Creating Vector Store ===")
    print(f"Total chunks to process: {total_chunks}")
    print("\nStarting vector store creation...")
    print("Note: This process may take several minutes.")
    
    try:
        # Create vector store without any extra parameters
        vector_store = FAISS.from_documents(
            documents=split_docs,
            embedding=embeddings
        )
        
        print("\nVector store created successfully!")
        return vector_store
            
    except Exception as e:
        print(f"\nError during vector store creation: {str(e)}")
        return None

def main():
    """Main function to set up the vector store with document embeddings"""
    print("=== Starting Vector Store Setup ===")
    
    # Check if API key is present in environment
    api_key = os.getenv("TOGETHER_API_KEY")
    print(f"API Key present: {bool(api_key)}")
    
    # Verify document directory and list available PDFs
    docs_path = Path("src/default_docs")
    print(f"Documents directory exists: {docs_path.exists()}")
    print(f"Documents found: {[pdf_file.name for pdf_file in docs_path.glob('*.pdf')]}")
    
    # Initialize the embedding model
    try:
        embeddings = TogetherEmbeddings(
            model="togethercomputer/m2-bert-80M-8k-retrieval",
            together_api_key=api_key
        )
        print("Embeddings initialized successfully")
    except Exception as e:
        print(f"Error initializing embeddings: {e}")
        return
        
    # Load and process all PDF documents
    documents = []
    with tqdm(total=len(list(docs_path.glob("*.pdf"))), desc="Processing PDFs") as pbar:
        for pdf_file in docs_path.glob("*.pdf"):
            print(f"\nProcessing: {pdf_file.name}")
            try:
                # Load PDF file
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                # Clean text content for each document
                for doc in docs:
                    doc.page_content = preprocess_text(doc.page_content)
                documents.extend(docs)
                print(f"- Loaded {len(docs)} pages")
            except Exception as e:
                print(f"- Error: {e}")
            pbar.update(1)
    
    print(f"\nTotal documents loaded: {len(documents)}")
    
    # Process documents if any were loaded successfully
    if documents:
        # Configure text splitting parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        # Split documents into smaller chunks
        split_docs = text_splitter.split_documents(documents)
        print(f"Created {len(split_docs)} text chunks")
        
        vector_store = create_vector_store(split_docs, embeddings)
        if vector_store:
            print("\nSaving vector store...")
            vector_store.save_local(
                folder_path="faiss_index",
                index_name="default_index"
            )
            print(f"Vector store saved successfully to {os.path.abspath('faiss_index')}")
        else:
            print("Failed to create vector store")
    else:
        print("No documents to process!")

# Entry point of the script
if __name__ == "__main__":
    main()