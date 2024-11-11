import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_together import TogetherEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def main():
    print("=== Starting Vector Store Setup ===")
    
    # Check environment
    api_key = os.getenv("TOGETHER_API_KEY")
    print(f"API Key present: {bool(api_key)}")
    
    # Check paths
    docs_path = Path("src/default_docs")
    print(f"Documents directory exists: {docs_path.exists()}")
    print(f"Documents found: {list(docs_path.glob('*.pdf'))}")
    
    # Initialize embeddings
    try:
        embeddings = TogetherEmbeddings(
            model="togethercomputer/m2-bert-80M-8k-retrieval",
            together_api_key=api_key
        )
        print("Embeddings initialized successfully")
    except Exception as e:
        print(f"Error initializing embeddings: {e}")
        return
        
    # Process documents
    documents = []
    for pdf_file in docs_path.glob("*.pdf"):
        print(f"\nProcessing: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            print(f"- Loaded {len(docs)} pages")
            documents.extend(docs)
        except Exception as e:
            print(f"- Error: {e}")
    
    print(f"\nTotal documents loaded: {len(documents)}")
    
    # Split documents
    if documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)
        print(f"Created {len(split_docs)} text chunks")
        
        print("\nCreating vector store...")
        print("This may take a few minutes depending on the number of chunks...")
        try:
            vector_store = FAISS.from_documents(
                documents=split_docs,
                embedding=embeddings
            )
            print("Vector store created!")
            
            print("\nSaving vector store...")
            vector_store.save_local(
                folder_path="faiss_index",
                index_name="default_index"
            )
            print(f"Vector store saved successfully to {os.path.abspath('faiss_index')}")
            print("\n=== Setup Complete ===")
        except Exception as e:
            print(f"Error creating/saving vector store: {e}")
    else:
        print("No documents to process!")

if __name__ == "__main__":
    main()