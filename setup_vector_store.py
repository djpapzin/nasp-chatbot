import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_together import TogetherEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize embeddings
    embeddings = TogetherEmbeddings(
        model="togethercomputer/m2-bert-80M-8k-retrieval",
        together_api_key=os.getenv("TOGETHER_API_KEY")
    )
    
    # Set up paths
    index_path = Path("faiss_index")
    docs_path = Path("src/default_docs")
    index_path.mkdir(exist_ok=True)
    
    # Process local PDF files
    print("Processing PDF files...")
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    for pdf_file in docs_path.glob("*.pdf"):
        print(f"Processing {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents.extend(loader.load())
            print(f"Successfully processed {pdf_file.name}")
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {str(e)}")
    
    # Split documents
    if not documents:
        print("No documents were processed successfully!")
        return
        
    split_docs = text_splitter.split_documents(documents)
    print(f"Creating vector store from {len(split_docs)} document chunks...")
    
    # Create and save vector store
    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local(folder_path=str(index_path), index_name="default_index")
    
    print("Vector store created successfully!")

if __name__ == "__main__":
    main()