import os
import logging
from pathlib import Path
from typing import Tuple, Any, List, Dict, Set
from datetime import datetime

# Telegram imports
from telegram import Update
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
from config import CHATBOT_CONFIG

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
    
    # Group documents by source
    sources = {}
    for doc in source_docs:
        source = doc.metadata.get('source', 'Unknown Source')
        page = doc.metadata.get('page', 1)
        
        # Clean up source name
        source = os.path.basename(source)  # Remove path
        source = os.path.splitext(source)[0]  # Remove extension
        source = source.replace('_', ' ')  # Replace underscores with spaces
        
        if source not in sources:
            sources[source] = set()
        sources[source].add(page)
    
    # Format the sources text
    sources_text = "\n\n📚 Sources:\n"
    for source, pages in sources.items():
        pages_str = f"(Page{'s' if len(pages) > 1 else ''} {', '.join(map(str, sorted(pages)))})"
        sources_text += f"- {source} {pages_str}\n"
    
    return sources_text

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming messages."""
    try:
        question = update.message.text
        chat_id = update.message.chat_id
        
        # Send typing indicator
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        
        # Generate response using RAG
        response = qa_chain.invoke({
            "input": question
        })
        
        if response and "answer" in response:
            # Send the main answer with markdown formatting
            await context.bot.send_message(
                chat_id=chat_id,
                text=response["answer"],
                parse_mode="Markdown"
            )
            
            # If there are source documents, format and send them
            if "source_documents" in response:
                sources_text = format_sources(response["source_documents"])
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=sources_text,
                    parse_mode="Markdown"
                )
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text="I'm sorry, I couldn't find relevant information to answer your question.",
                parse_mode="Markdown"
            )
    except Exception as e:
        logger.error(f"Error handling message: {str(e)}", exc_info=True)
        await context.bot.send_message(
            chat_id=chat_id,
            text="I encountered an error while processing your request. Please try again.",
            parse_mode="Markdown"
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
    """Send a message when the command /start is issued."""
    user = update.effective_user
    welcome_message = (
        f"👋 Hi {user.mention_html()}!\n\n"
        f"{CHATBOT_CONFIG['description']}\n\n"
        "Available commands:\n"
        "/start - Show this welcome message\n"
        "/help - Show help information\n\n"
        f"{CHATBOT_CONFIG['opening_message']}"
    )
    await update.message.reply_html(welcome_message)

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
        ("system", SYSTEM_PROMPT),
        ("human", "{input}")
    ])
    
    document_chain = create_stuff_documents_chain(
        llm, 
        prompt,
        document_variable_name="context"
    )
    
    # Create retrieval chain
    return create_retrieval_chain(
        retriever, 
        document_chain,
        return_source_documents=True
    )

# Constants
SYSTEM_PROMPT = """Your task is to be an expert researcher that can answer questions. Use the following pieces of retrieved context to answer the question. Use clear and professional language, and organize the summary in a logical manner using appropriate formatting such as headings, subheadings, and bullet points. If the information needed to answer the question is not available in the context then say that you don't know.

Context: {context}
"""

def main() -> None:
    """Start the bot"""
    try:
        # Initialize components
        global vector_store, vector_search, qa_chain
        vector_store, vector_search, qa_chain = init_components()
        
        # Create application
        application = Application.builder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
        
        # Start bot
        logger.info("Starting bot...")
        application.run_polling()
        
    except Exception as e:
        logger.error(f"Fatal error starting bot: {str(e)}")
        raise

if __name__ == '__main__':
    main()
