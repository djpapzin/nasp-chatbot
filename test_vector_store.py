import os
from dotenv import load_dotenv
from langchain_together import TogetherEmbeddings
from langchain_community.vectorstores import FAISS

def test_vector_store():
    """Test vector store retrieval"""
    print("\n=== Testing Vector Store ===")
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("TOGETHER_API_KEY")
    
    try:
        # Initialize embeddings
        print("Initializing embeddings...")
        embeddings = TogetherEmbeddings(
            model="togethercomputer/m2-bert-80M-8k-retrieval",
            together_api_key=api_key
        )
        
        # Load vector store
        print("Loading vector store...")
        vector_store = FAISS.load_local(
            folder_path="faiss_index",
            embeddings=embeddings,
            index_name="default_index",
            allow_dangerous_deserialization=True
        )
        print(f"Vector store loaded with {len(vector_store.docstore._dict)} documents")
        
        # Test queries with similarity scores
        test_queries = [
            "What does the SPIL document say about social protection?",
            "What are the main findings about unpaid care work in Uzbekistan?",
            "Tell me about employment services for persons with disabilities",
        ]
        
        for query in test_queries:
            print(f"\n\nQuery: {query}")
            
            # Use similarity_search_with_score instead
            docs_and_scores = vector_store.similarity_search_with_score(
                query,
                k=4,  # Get more results
                fetch_k=20  # Fetch more candidates
            )
            
            print(f"Found {len(docs_and_scores)} results:")
            
            for i, (doc, score) in enumerate(docs_and_scores, 1):
                print(f"\nResult {i} (Score: {score:.4f}):")
                print(f"Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))}")
                print(f"Page: {doc.metadata.get('page', 'Unknown')}")
                print(f"Content preview: {doc.page_content[:200]}...")
    
    except Exception as e:
        print(f"Error during testing: {str(e)}")

if __name__ == "__main__":
    test_vector_store()