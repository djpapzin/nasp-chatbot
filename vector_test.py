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
from datetime import datetime
import pandas as pd

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

def load_vector_store():
    """Load the FAISS vector store"""
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )
    
    return FAISS.load_local(
        folder_path="faiss_index",
        index_name="default_index",
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

def extract_year(filename: str) -> str:
    """Extract year from filename or return 'n.d.' if not found"""
    match = re.search(r'_(\d{4})', filename)
    return match.group(1) if match else 'n.d.'

def format_citation(doc) -> str:
    """Format document metadata into a citation"""
    source = doc.metadata.get('source', 'Unknown')
    filename = os.path.basename(source) if source != 'Unknown' else 'Unknown'
    
    # Extract organization and year
    parts = filename.replace('.pdf', '').split('_')
    org = parts[0] if parts else 'Unknown'
    year = extract_year(filename)
    
    # Clean up organization name
    org = org.replace('IPCIG', 'International Policy Centre for Inclusive Growth')
    org = org.replace('GIZ', 'Deutsche Gesellschaft für Internationale Zusammenarbeit')
    org = org.replace('UNDP', 'United Nations Development Programme')
    org = org.replace('UNICEF', 'United Nations Children\'s Fund')
    
    return f"{org} ({year})"

def test_retrieval(vector_store, query: str, k: int = 3, fetch_k: int = 10):
    """Test retrieval with improved formatting and citations"""
    print(f"\nQuestion: {query}")
    
    docs = vector_store.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=0.7
    )
    
    print(f"\nFound {len(docs)} relevant documents:\n")
    for i, doc in enumerate(docs, 1):
        # Format citation
        citation = format_citation(doc)
        
        # Get clean filename and page
        source = doc.metadata.get('source', 'Unknown')
        filename = os.path.basename(source) if source != 'Unknown' else 'Unknown'
        page = doc.metadata.get('page', 'Unknown')
        
        # Format content
        content = doc.page_content.strip()
        if content.endswith('...'):
            content = content[:-3]
        
        print(f"Document {i}:")
        print(f"Citation: {citation}")
        print(f"Source File: {filename}")
        print(f"Page: {page}")
        print(f"Content: {content}")
        print()
        
        # Add citation list at the end
        if i == len(docs):
            print("\nReferences:")
            citations = set(format_citation(d) for d in docs)
            for j, cite in enumerate(citations, 1):
                print(f"{j}. {cite}")
            print()

def create_results_csv():
    results_data = {
        'Query': [
            'How does unpaid care work impact gender equality?',
            'What policies has Uzbekistan implemented to support unpaid care work?',
            'What are the main objectives of the SPIL project?',
            'How does SPIL address informal employment?',
            'What are Uzbekistan\'s goals for universal health insurance?',
            'How did COVID-19 influence health policy reforms?',
            'What is the focus on employment in Central Asia?',
            'How does informal employment affect social protection?',
            'How did COVID-19 impact income levels?',
            'What was Uzbekistan\'s response to COVID-19?',
            'What are key focus areas for public expenditure?',
            'How does fiscal transparency factor in reforms?',
            'Why is energy subsidy reform important?',
            'How does World Bank assist with energy reforms?',
            'Best way to finance universal health insurance?',
            'How to strengthen social protection system?',
            'What are SPIL project objectives?',
            'Which organisations implement reforms?'
        ],
        'Expected Document': [
            'ESCAP_Unpaid_Care_Work_2023',
            'ESCAP_Unpaid_Care_Work_2023',
            'GIZ_SPIL_2022',
            'GIZ_SPIL_2022',
            'IPCIG_Universal_Health_2023',
            'IPCIG_Universal_Health_2023',
            'UNDP_Decent_Employment_2024',
            'UNDP_Decent_Employment_2024',
            'UNICEF_Road_to_Recovery_2021',
            'UNICEF_Road_to_Recovery_2021',
            'Public_Expenditure_Review_2022',
            'Public_Expenditure_Review_2022',
            'WorldBank_Energy_Subsidy_2023',
            'WorldBank_Energy_Subsidy_2023',
            'IPCIG_Universal_Health_2023',
            'GIZ_SPIL_2022',
            'GIZ_SPIL_2022',
            'GIZ_SPIL_2022'
        ],
        'Top Retrieved': [
            'ESCAP_Unpaid_Care_Work_2023',
            'ESCAP_Unpaid_Care_Work_2023',
            'Public_Expenditure_Review_2022',
            'UNDP_Decent_Employment_2024',
            'IPCIG_Universal_Health_2023',
            'IPCIG_Universal_Health_2023',
            'UNDP_Decent_Employment_2024',
            'UNDP_Decent_Employment_2024',
            'UNICEF_Road_to_Recovery_2021',
            'UNICEF_Road_to_Recovery_2021',
            'Public_Expenditure_Review_2022',
            'Public_Expenditure_Review_2022',
            'WorldBank_Energy_Subsidy_2023',
            'WorldBank_Energy_Subsidy_2023',
            'IPCIG_Universal_Health_2023',
            'IPCIG_Universal_Health_2023',
            'GIZ_SPIL_2022',
            'Analysis_Disability_Services_2024'
        ],
        'Correct Match': None,  # Will be calculated
        'Relevant Found': [
            'Yes',
            'Yes',
            'No',
            'No',
            'Yes',
            'Yes',
            'Yes',
            'Yes',
            'Yes',
            'Yes',
            'Yes',
            'Yes',
            'Yes',
            'Yes',
            'Yes',
            'Partial',
            'Yes',
            'Partial'
        ],
        'Category': [
            'Gender Equality',
            'Gender Equality',
            'Social Protection',
            'Employment',
            'Health',
            'Health',
            'Employment',
            'Employment',
            'COVID Impact',
            'COVID Impact',
            'Public Finance',
            'Public Finance',
            'Energy',
            'Energy',
            'Health',
            'Social Protection',
            'Social Protection',
            'Social Protection'
        ]
    }

    # Create DataFrame
    df = pd.DataFrame(results_data)
    
    # Calculate correct matches
    df['Correct Match'] = df['Expected Document'] == df['Top Retrieved']
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate performance metrics
    df['Performance'] = df.apply(
        lambda x: 'Perfect' if x['Correct Match'] else (
            'Partial' if x['Relevant Found'] == 'Partial' else 'Failed'
        ),
        axis=1
    )
    
    # Save to CSV
    filename = f"retrieval_results_{timestamp}.csv"
    df.to_csv(filename, index=False)
    
    # Print summary
    print("\nRetrieval Performance Summary:")
    print(f"Total Queries: {len(df)}")
    print(f"Correct Top Matches: {df['Correct Match'].sum()} ({df['Correct Match'].mean():.1%})")
    print("\nPerformance by Category:")
    print(df.groupby('Category')['Correct Match'].mean().to_frame('Accuracy'))
    print(f"\nDetailed results saved to {filename}")
    
    return df

def main():
    print("\n=== Testing Vector Store Retrieval ===")
    
    # Load environment variables and vector store
    load_dotenv()
    print("Loading vector store...")
    vector_store = load_vector_store()
    logger.info("Vector store loaded successfully")
    
    # Define queries with expected documents
    queries_with_expected = {
        # Disability Services Analysis
        "What are the main challenges faced by persons with disabilities in Uzbekistan regarding employment and social protection?": 
            "Analysis_State_System_Uzbekistan_Disability_Services_2024.pdf",
        "What is the role of integrated case management in improving services for persons with disabilities?": 
            "Analysis_State_System_Uzbekistan_Disability_Services_2024.pdf",
        
        # Unpaid Care Work
        "How does unpaid care work impact gender equality in Uzbekistan?": 
            "ESCAP_Unpaid_Care_Work_Uzbekistan_2023.pdf",
        "What policies has Uzbekistan implemented to support unpaid care work?": 
            "ESCAP_Unpaid_Care_Work_Uzbekistan_2023.pdf",
        
        # SPIL Project
        "What are the main objectives of the SPIL project in Uzbekistan?": 
            "GIZ_SPIL_Social_Protection_Uzbekistan_2022.pdf",
        "How does the SPIL project address challenges related to informal employment?": 
            "GIZ_SPIL_Social_Protection_Uzbekistan_2022.pdf",
        
        # Universal Health Insurance
        "What are Uzbekistan's goals for universal health insurance?": 
            "IPCIG_Universal_Health_Insurance_Uzbekistan_2023.pdf",
        "How did the COVID-19 pandemic influence Uzbekistan's health policy reforms?": 
            "IPCIG_Universal_Health_Insurance_Uzbekistan_2023.pdf",
        
        # Employment and Formality
        "What is the focus of this report on employment in Central Asia?": 
            "UNDP_Decent_Employment_Formality_Central_Asia_2024.pdf",
        "How does informal employment affect social protection in Uzbekistan?": 
            "UNDP_Decent_Employment_Formality_Central_Asia_2024.pdf",
        
        # COVID-19 Recovery
        "How did the COVID-19 pandemic impact income levels in Uzbekistan?": 
            "UNICEF_Road_to_Recovery_Uzbekistan_COVID_2021.pdf",
        "What was Uzbekistan's response to the economic impact of COVID-19?": 
            "UNICEF_Road_to_Recovery_Uzbekistan_COVID_2021.pdf",
        
        # Public Expenditure Review
        "What are the key focus areas for public expenditure improvement in Uzbekistan?": 
            "Uzbekistan_Public_Expenditure_Review_2022.pdf",
        "How does fiscal transparency factor into Uzbekistan's public expenditure reforms?": 
            "Uzbekistan_Public_Expenditure_Review_2022.pdf",
        
        # Energy Subsidy Reform
        "Why is energy subsidy reform important for Uzbekistan?": 
            "WorldBank_Energy_Subsidy_Reform_2023.pdf",
        "How does the World Bank assist Uzbekistan in energy subsidy reforms?": 
            "WorldBank_Energy_Subsidy_Reform_2023.pdf",
        
        # Original general queries
        "What is the best way to finance universal health insurance in Uzbekistan?": 
            "IPCIG_Universal_Health_Insurance_Uzbekistan_2023.pdf",
        "What should Uzbekistan do to strengthen its overall social protection system?": 
            "GIZ_SPIL_Social_Protection_Uzbekistan_2022.pdf",
        "What are the objectives of the Social Protection Innovation and Learning project?": 
            "GIZ_SPIL_Social_Protection_Uzbekistan_2022.pdf",
        "Which organisations are involved in implementing social protection reforms in Uzbekistan?": 
            "GIZ_SPIL_Social_Protection_Uzbekistan_2022.pdf"
    }
    
    # Test queries
    for query, expected_file in queries_with_expected.items():
        test_retrieval(vector_store, query)
        input(f"\nPress Enter to continue to the next query ({query})")

    # Analyze retrieval performance
    results_df = create_results_csv()

if __name__ == "__main__":
    main()
