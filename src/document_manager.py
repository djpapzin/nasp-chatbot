import os
import requests
import tempfile
from pathlib import Path
from typing import List, Tuple, Union, BinaryIO
import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader

logger = logging.getLogger(__name__)

class DocumentManager:
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
            for doc in DocumentManager.DEFAULT_DOCS:
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

    def process_file(self, file: Union[str, BinaryIO], filename: str = None) -> Tuple[bool, Union[List, str]]:
        """
        Process a file from either Streamlit upload or Telegram
        
        Args:
            file: Either a file path (str) or file-like object
            filename: Original filename (required for file-like objects)
            
        Returns:
            Tuple[bool, Union[List, str]]: (success, result)
                - If success is True, result is list of documents
                - If success is False, result is error message
        """
        try:
            # Handle file path
            if isinstance(file, str):
                return True, self.load_document(file)
            
            # Handle file-like object (Streamlit or Telegram)
            if not filename:
                return False, "Filename is required for uploaded files"
                
            file_extension = filename.split('.')[-1].lower()
            if file_extension not in ['pdf', 'txt', 'docx']:
                return False, "Unsupported file type. Please upload PDF, DOCX, or TXT files."
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as temp_file:
                # Handle Streamlit upload
                if hasattr(file, 'getbuffer'):
                    temp_file.write(file.getbuffer())
                # Handle Telegram file
                else:
                    temp_file.write(file.read())
                
                docs = self.load_document(temp_file.name)
                os.unlink(temp_file.name)
                
                if not docs:
                    return False, "No content could be extracted from the file"
                    
                return True, docs
                
        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")
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