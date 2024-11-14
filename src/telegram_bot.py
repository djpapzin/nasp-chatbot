import os
import logging
from pathlib import Path
from typing import Tuple, Any, List, Dict, Set
from datetime import datetime
import re

# Telegram imports
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_together import TogetherEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings

# Local imports
from document_manager import TelegramDocumentManager
from vector_search import VectorSearch
from config import PROMPT_CONFIG, CHATBOT_CONFIG

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler(), logging.FileHandler('bot.log')]
)
logger = logging.getLogger(__name__)

# Disable httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# Session storage - using a simple dict for in-memory storage
user_sessions: Dict[int, Set[str]] = {}  # user_id -> set of uploaded document names

def format_sources(source_docs: List[Document]) -> str:
    """Format source documents into a readable string."""
    if not source_docs:
        return ""
    
    sources_text = "\n\n📚 *View Sources*\n\n"
    
    for doc in source_docs:
        # Get source and page info
        source = doc.metadata.get('source', 'Unknown Source')
        page = doc.metadata.get('page', 'N/A')
        
        # Clean up source path to match Streamlit format
        source = source.replace('src\\default_docs\\', '')  # Remove path prefix
        
        # Format source header
        source_header = f"📄 *Source:* {escape_markdown(source)} 📑 *Pages:* {page}\n\n"
        
        # Format relevant excerpts
        content = doc.page_content.strip()
        excerpts = f"🔍 *Relevant Excerpts:*\n{escape_markdown(content)}\n\n"
        
        sources_text += source_header + excerpts
    
    return sources_text

def escape_markdown(text: str) -> str:
    """Escape special characters in Markdown."""
    escape_chars = r"[\\*_\[\]()~`>#+\-=|{}.!]"  # Add other needed chars
    return re.sub(escape_chars, lambda match: "\\" + match.group(0), text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming messages."""
    try:
        query = update.message.text
        logger.info(f"Received query: {query}")
        
        await update.message.chat.send_action(action="typing")
        
        logger.info("Invoking LLM chain...")
        result = await qa_chain.ainvoke({"input": query})
        logger.info("LLM chain completed")
        
        answer = result.get('answer', 'No answer generated')
        sources = format_sources(result.get('source_documents', []))
        
        # Combine answer and sources, escaping special characters
        full_response = f"{escape_markdown(answer)}{sources}"
        
        await update.message.reply_text(
            full_response,
            parse_mode=ParseMode.MARKDOWN_V2,
            disable_web_page_preview=True
        )
        logger.info("Response sent successfully")
        
    except Exception as e:
        logger.exception(f"Detailed error in handle_message: {str(e)}")
        error_message = escape_markdown(f"I encountered an error while processing your request: {str(e)}. Please try again.")
        await update.message.reply_text(
            error_message, 
            parse_mode=ParseMode.MARKDOWN_V2
        )

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle document messages."""
    try:
        # Get basic info about the document
        doc = update.message.document
        file_name = doc.file_name
        
        await update.message.reply_text(
            f"I received your document: {file_name}\n"
            "However, document processing is currently not implemented in the Telegram interface. "
            "Please use the web interface to upload and process documents."
        )
    except Exception as e:
        logger.error(f"Error handling document: {str(e)}")
        await update.message.reply_text(
            "Sorry, I encountered an error while processing your document. "
            "Please try again later or contact support if the issue persists."
        )

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send welcome message when /start command is issued."""
    welcome_message = (
        "*NASP Chatbot*\n\n"  # Title in bold
        "Welcome\\! This is a prototype chatbot for the National Agency of Social Protection\\. "
        "You can use it to ask questions about a library of reports, evaluations, research and other documents\\.\n\n"
        "Hello\\. Please enter your question in the chat box to get started\\."
    )
    
    await update.message.reply_text(
        welcome_message,
        parse_mode=ParseMode.MARKDOWN_V2
    )

def init_components():
    """Initialize LLM, vector store, and QA chain components"""
    logger.info("=== Starting Bot Initialization ===")
    
    try:
        # Initialize embeddings with Azure OpenAI
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
        logger.info("Embeddings initialized successfully")

        # Initialize LLM
        llm = ChatOpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=os.getenv("TOGETHER_API_KEY"),
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            temperature=0.7,
            max_tokens=1000
        )
        logger.info("LLM initialized successfully")

        # Initialize VectorSearch
        vector_search = VectorSearch()
        vector_search.embeddings = embeddings
        vector_search.vector_store = FAISS.load_local(
            folder_path="faiss_index",
            embeddings=embeddings,
            index_name="default_index",
            allow_dangerous_deserialization=True
        )
        logger.info("Vector store loaded successfully")

        # Create QA chain
        qa_chain = create_qa_chain(llm, vector_search)
        
        logger.info("=== Bot Initialization Complete ===")
        return vector_search.vector_store, vector_search, qa_chain
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise

def create_qa_chain(llm, vector_search):
    """Create the question-answering chain"""
    # Create retrieval chain
    retriever = vector_search.vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Create document chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPT_CONFIG["en"]["system_prompt"]),
        ("human", "{input}")
    ])
    
    document_chain = create_stuff_documents_chain(
        llm, 
        prompt,
        document_variable_name="context"
    )
    
    # Create retrieval chain with updated parameter names
    return create_retrieval_chain(
        retriever=retriever,  # Changed from vector_search to retriever
        combine_docs_chain=document_chain  # Changed from question_answer_chain to combine_docs_chain
    )

# Constants
SYSTEM_PROMPT = """Your task is to be an expert researcher that can answer questions. Use the following pieces of retrieved context to answer the question. Use clear and professional language, and organize the summary in a logical manner using appropriate formatting such as headings, subheadings, and bullet points. If the information needed to answer the question is not available in the context then say that you don't know.

Context: {context}
"""

if __name__ == "__main__":
    # Initialize components
    global vector_store, vector_search, qa_chain  # If these are needed globally
    vector_store, vector_search, qa_chain = init_components()

    # Create application and add handlers
    application = Application.builder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    # Start the bot.  NO asyncio.run() here!
    application.run_polling()
