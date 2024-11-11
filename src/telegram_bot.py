import os
import logging
import sys
from pathlib import Path
from typing import Tuple, Any
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_together import Together
from src.document_manager import DocumentManager
from src.vector_search import VectorSearch
from langchain.llms import Together as TogetherLLM
from io import BytesIO

# Configure logging with more detail
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log')
    ]
)
logger = logging.getLogger(__name__)

def init_components() -> Tuple[Any, Any, Any, Any]:
    """Initialize bot components with detailed logging"""
    try:
        logger.info("=== Starting Bot Initialization ===")
        
        # Stage 1: Initialize Document Manager
        logger.info("Stage 1: Initializing Document Manager")
        doc_manager = DocumentManager()
        
        # Stage 2: Load Default Documents
        logger.info("Stage 2: Loading Default Documents")
        default_docs = []
        for doc in doc_manager.DEFAULT_DOCS:
            logger.info(f"Processing document: {doc['name']}")
            docs = doc_manager.load_document_from_url(doc['url'])
            default_docs.extend(docs)
        logger.info(f"Loaded {len(default_docs)} document chunks")
        
        # Stage 3: Initialize Vector Store
        logger.info("Stage 3: Creating Vector Store")
        vector_search = VectorSearch()
        vector_store, _ = vector_search.create_vector_store(default_docs)
        logger.info("Vector store created successfully")
        
        # Stage 4: Initialize LLM
        logger.info("Stage 4: Initializing LLM")
        Together().api_key = os.getenv("TOGETHER_API_KEY")
        llm = TogetherLLM(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            temperature=0.7,
            max_tokens=1000
        )
        
        # Stage 5: Setup Memory and QA Chain
        logger.info("Stage 5: Setting up Memory and QA Chain")
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            verbose=True
        )
        
        logger.info("=== Bot Initialization Complete ===")
        return vector_store, llm, qa_chain, doc_manager
        
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}", exc_info=True)
        raise

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    logger.info(f"New user started bot: {update.effective_user.id}")
    welcome_message = (
        "Welcome! This is a prototype chatbot for the National Agency of Social Protection. "
        "You can use it to ask questions about a library of reports, evaluations, research and other documents.\n\n"
        "The following documents are available:\n"
        "• Exploring Pathways to Decent Employment in Central Asia\n"
        "• Social Protection Innovation and Learning in Uzbekistan\n"
        "• Uzbekistan Public Expenditure Review\n"
        "• Valuing and investing in unpaid care in Uzbekistan\n"
        "• Prioritising universal health insurance in Uzbekistan\n\n"
        "Hello. Please enter your question in the chat box to get started."
    )
    await update.message.reply_text(welcome_message)

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle document uploads with logging"""
    user_id = update.effective_user.id
    file_name = update.message.document.file_name
    logger.info(f"User {user_id} uploaded document: {file_name}")
    
    try:
        await update.message.reply_text("📄 Processing your document...")
        
        # Get file from Telegram
        file = await update.message.document.get_file()
        file_content = await file.download_as_bytearray()
        
        # Process using centralized handler
        success, result = doc_manager.process_file(
            BytesIO(file_content),
            filename=file_name
        )
        
        if success:
            logger.info(f"Successfully processed document: {file_name}")
            vector_store.add_documents(result)
            await update.message.reply_text(
                "✅ Document added to the library!\n"
                "You can now ask questions about its content."
            )
        else:
            logger.warning(f"Failed to process document: {file_name}. Error: {result}")
            await update.message.reply_text(
                f"❌ Sorry, I couldn't process your document.\n"
                f"Error: {result}"
            )
            
    except Exception as e:
        logger.error(f"Error processing document {file_name}: {str(e)}", exc_info=True)
        await update.message.reply_text(
            "Sorry, I encountered an error processing your document. "
            "Please try again or contact support."
        )

def main():
    """Start the bot with initialization logging"""
    try:
        logger.info("Starting NASP Chatbot")
        
        # Load environment variables
        load_dotenv()
        if not os.getenv("TELEGRAM_BOT_TOKEN"):
            logger.error("TELEGRAM_BOT_TOKEN not found in environment variables")
            raise ValueError("TELEGRAM_BOT_TOKEN not found")
            
        if not os.getenv("TOGETHER_API_KEY"):
            logger.error("TOGETHER_API_KEY not found in environment variables")
            raise ValueError("TOGETHER_API_KEY not found")
        
        # Initialize components
        logger.info("Initializing bot components")
        global vector_store, llm, qa_chain, doc_manager
        vector_store, llm, qa_chain, doc_manager = init_components()
        
        # Create application
        application = Application.builder().token(os.getenv('TELEGRAM_BOT_TOKEN')).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        application.add_handler(MessageHandler(filters.DOCUMENT, handle_document))
        
        logger.info("Bot is ready to start polling")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
        
    except Exception as e:
        logger.error("Fatal error starting bot", exc_info=True)
        raise

if __name__ == '__main__':
    main()
