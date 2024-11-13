import os
import requests
import tempfile
from pathlib import Path
from typing import List, Tuple, Union, BinaryIO
import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from datetime import datetime
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

# Create a base document manager without Streamlit dependencies
class BaseDocumentManager:
    DEFAULT_DOCS = [
        {
            "name": "Exploring Pathways to Decent Employment",
            "url": "https://www.undp.org/sites/g/files/zskgke326/files/2024-07/report_-_ii_-_informality_-_22072024_-_final_iii.pdf"
        },
        {
            "name": "Social Protection Innovation and Learning",
            "url": "https://www.giz.de/de/downloads/giz2022-en-SPIL-uzbekistan.pdf"
        },
        {
            "name": "Uzbekistan Public Expenditure Review",
            "url": "https://openknowledge.worldbank.org/server/api/core/bitstreams/2b51e43c-663d-4e0f-8afe-b6b682815ed3/content"
        },
        {
            "name": "Valuing and investing in unpaid care",
            "url": "https://repository.unescap.org/bitstream/handle/20.500.12870/5433/ESCAP-2023-RP-Unpaid-Care-Domestic-Work-Uzbekistan-E.pdf"
        },
        {
            "name": "Prioritising universal health insurance",
            "url": "https://ipcig.org/sites/default/files/pub/en/OP502_Prioritising_universal_health_insurance_in_Uzbekistan.pdf"
        }
    ]

    def __init__(self):
        """Initialize document manager"""
        self.index_path = Path("faiss_index")
        self.index_path.mkdir(exist_ok=True)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    @staticmethod
    def load_default_documents():
        """Load and process default documents"""
        try:
            documents = []
            for doc in BaseDocumentManager.DEFAULT_DOCS:
                logger.info(f"Processing default document: {doc['name']}")
                response = requests.get(doc['url'])
                response.raise_for_status()
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(response.content)
                    temp_file.flush()  # Ensure all data is written
                    docs = PyPDFLoader(temp_file.name).load()
                    documents.extend(docs)
                    try:
                        os.unlink(temp_file.name)
                    except Exception as e:
                        logger.warning(f"Could not delete temp file: {e}")
            
            return documents
        except Exception as e:
            logger.error(f"Error loading default documents: {str(e)}")
            return []

    def process_file(self, file, source_name=None) -> Tuple[bool, Union[List[Document], str]]:
        """Process a file and return a tuple of (success, result)
        where result is either a list of Documents or an error message"""
        try:
            # Handle Streamlit UploadedFile
            if hasattr(file, 'getvalue'):
                # Save uploaded file temporarily
                temp_path = f"temp_{source_name or file.name}"
                with open(temp_path, "wb") as f:
                    f.write(file.getvalue())
                file_path = temp_path
            else:
                # Handle regular file path
                file_path = file

            # Process based on file type
            if file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.lower().endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            elif file_path.lower().endswith('.txt'):
                loader = TextLoader(file_path)
            else:
                return False, f"Unsupported file type: {file_path}"

            # Load and split documents
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)

            # Add source metadata if not present
            for doc in splits:
                if 'source' not in doc.metadata and source_name:
                    doc.metadata['source'] = source_name

            # Clean up temp file if it was created
            if hasattr(file, 'getvalue') and os.path.exists(temp_path):
                os.remove(temp_path)

            return True, splits

        except Exception as e:
            logger.error(f"Error processing file: {str(e)}", exc_info=True)
            return False, str(e)

    def load_document(self, file_path: str) -> List:
        """Load a single document"""
        try:
            if file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.lower().endswith('.txt'):
                loader = TextLoader(file_path)
            elif file_path.lower().endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
            
            return loader.load()
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            return []

    def load_default_documents(self):
        return []

# Create a UI-specific document manager that inherits from base
class WebDocumentManager(BaseDocumentManager):
    def __init__(self):
        super().__init__()
        import streamlit as st  # Only import streamlit here
        self.st = st
        # ... UI specific functionality ...

# Create a Telegram-specific document manager
class TelegramDocumentManager(BaseDocumentManager):
    # ... telegram specific functionality ...
    pass