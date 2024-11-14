import os
import logging
from pathlib import Path
from typing import Tuple, Any, List, Dict, Set
from datetime import datetime
import re
import asyncio

# Telegram imports
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram import BotCommand

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
from src.translator import Translator

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
    """Handle document uploads"""
    await update.message.reply_text(
        "Document uploads are not supported yet. Please check back later."
    )

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send welcome message when /start command is issued."""
    # Default to English initially
    user_language = context.user_data.get('language', 'en')
    
    # Initialize translator
    translator = Translator()
    
    # Get and translate welcome message
    welcome_message = CHATBOT_CONFIG['welcome_message']
    translated_welcome = translator.translate(welcome_message, user_language)
    
    # Escape special characters for Markdown
    escaped_message = escape_markdown(translated_welcome)
    
    # Add language selection instructions
    language_instructions = translator.translate(
        "Use /language to change the language (EN/RU/UZ)",
        user_language
    )
    
    full_message = f"{escaped_message}\n\n{escape_markdown(language_instructions)}"
    
    await update.message.reply_text(
        full_message,
        parse_mode=ParseMode.MARKDOWN_V2
    )

async def language_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle language change command"""
    translator = Translator()
    
    # Create keyboard with language options
    keyboard = [
        ["English 🇬🇧", "Русский 🇷🇺", "O'zbek 🇺🇿"]
    ]
    
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True)
    
    await update.message.reply_text(
        "Please select your language:\nПожалуйста, выберите язык:\nTilni tanlang:",
        reply_markup=reply_markup
    )

async def handle_language_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle language selection from keyboard"""
    text = update.message.text
    
    language_map = {
        "English 🇬🇧": "en",
        "Русский 🇷🇺": "ru",
        "O'zbek 🇺🇿": "uz"
    }
    
    if text in language_map:
        selected_language = language_map[text]
        context.user_data['language'] = selected_language
        
        translator = Translator()
        confirmation = translator.translate(
            "Language has been updated. You can now continue chatting.",
            selected_language
        )
        
        await update.message.reply_text(
            escape_markdown(confirmation),
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=ReplyKeyboardRemove()
        )

def init_components():
    """Initialize LLM, vector store, and QA chain components"""
    try:
        # Initialize embeddings
        vector_search = VectorSearch()
        vector_search.initialize_embeddings()
        
        # Load the same vector store as Streamlit
        vector_store = FAISS.load_local(
            folder_path="faiss_index",
            embeddings=vector_search.embeddings,
            index_name="index"
        )
        logger.info("Vector store loaded successfully")
        
        # Create QA chain
        qa_chain = create_qa_chain(llm, vector_store)
        
        return vector_store, vector_search, qa_chain
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise

def create_qa_chain(llm, vector_store):
    """Create the question-answering chain"""
    # Create retrieval chain
    retriever = vector_store.as_retriever(
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

async def set_commands(application: Application) -> None:
    """Set bot commands in Telegram UI."""
    commands = [
        ("start", "Start the bot and get welcome message"),
        ("help", "Show help information"),
        ("about", "Learn more about NASP Chatbot"),
        ("language", "Change language (EN/RU/UZ)"),
        ("clear_chat", "Clear chat history"),
    ]
    
    await application.bot.set_my_commands(
        [BotCommand(command, description) for command, description in commands]
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send help message when /help command is issued."""
    help_text = (
        "*Available Commands:*\n\n"
        "/start \\- Start the bot and get welcome message\n"
        "/help \\- Show this help message\n"
        "/about \\- Learn more about NASP Chatbot\n"
        "/language \\- Change language \\(EN/RU/UZ\\)\n"
        "/clear\\_chat \\- Clear chat history\n\n"
        "You can also simply type your question and I will try to answer it\\!"
    )
    
    await update.message.reply_text(
        help_text,
        parse_mode=ParseMode.MARKDOWN_V2
    )

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send information about the bot when /about command is issued."""
    about_text = (
        "*About NASP Chatbot*\n\n"
        "I am a chatbot designed to help you find information about social protection in Uzbekistan\\. "
        "I can answer questions based on various reports, evaluations, and research documents\\.\n\n"
        "I use AI technology to understand your questions and provide relevant information from official sources\\."
    )
    
    await update.message.reply_text(
        about_text,
        parse_mode=ParseMode.MARKDOWN_V2
    )

async def clear_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear chat history for the current user."""
    user_id = update.effective_user.id
    
    # Clear user's chat history if it exists
    if user_id in user_sessions:
        user_sessions[user_id] = set()
        await update.message.reply_text(
            escape_markdown("Chat history has been cleared\\!"),
            parse_mode=ParseMode.MARKDOWN_V2
        )
    else:
        await update.message.reply_text(
            escape_markdown("No chat history to clear\\!"),
            parse_mode=ParseMode.MARKDOWN_V2
        )

async def main() -> None:
    """Main function to run the bot."""
    try:
        # Initialize components
        global vector_store, vector_search, qa_chain
        vector_store, vector_search, qa_chain = init_components()

        # Create application and add handlers
        application = Application.builder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()
        
        # Add command handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("about", about_command))
        application.add_handler(CommandHandler("clear_chat", clear_chat))
        application.add_handler(CommandHandler("language", language_command))
        application.add_handler(MessageHandler(
            filters.Regex("^(English 🇬🇧|Русский 🇷🇺|O'zbek 🇺🇿)$"), 
            handle_language_selection
        ))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        application.add_handler(MessageHandler(filters.Document.ALL, handle_document))

        # Set up commands in Telegram UI
        await set_commands(application)

        # Initialize the application
        await application.initialize()

        # Start polling in the current event loop
        logger.info("Starting bot...")
        await application.updater.start_polling()
        await application.start()

        # Manually idle the bot
        while True:
            await asyncio.sleep(1)  # Check every second for shutdown

    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(main())  # Run the main function
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    except Exception as e:
        logger.error(f"Bot stopped due to error: {str(e)}", exc_info=True)
