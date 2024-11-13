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
from langchain.chat_models import ChatOpenAI
from langchain.document import Document

# Local imports
from document_manager import DocumentManager
from vector_search import VectorSearch

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

class TelegramBot:
    def __init__(self, token: str):
        self.application = Application.builder().token(token).build()
        self._setup_handlers()
        
    def _setup_handlers(self):
        """Set up command and message handlers"""
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("reset", self.reset_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        
        # Document handler
        self.application.add_handler(MessageHandler(filters.Document.ALL, self.handle_document))

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command - initialize or reset session"""
        user_id = update.effective_user.id
        user_sessions[user_id] = set()  # Initialize empty session
        
        welcome_message = (
            "👋 Welcome to the NASP Document Assistant!\n\n"
            "I can help you with questions about social protection documents.\n\n"
            "Commands:\n"
            "📤 Send me documents to analyze\n"
            "/status - See current session info\n"
            "/reset - Start a new session\n"
            "/start - Show this message\n\n"
            "Currently using default documents. Upload your own to add to the knowledge base."
        )
        await update.message.reply_text(welcome_message)

    async def reset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /reset command - clear session and return to defaults"""
        user_id = update.effective_user.id
        old_docs = len(user_sessions.get(user_id, set()))
        user_sessions[user_id] = set()
        
        reset_message = (
            "🔄 Session reset complete!\n\n"
            f"Removed {old_docs} uploaded documents.\n"
            "Returned to default documents only.\n"
            "You can start uploading new documents or ask questions."
        )
        await update.message.reply_text(reset_message)

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command - show current session info"""
        user_id = update.effective_user.id
        uploaded_docs = user_sessions.get(user_id, set())
        
        status_message = [
            "📊 Current Session Status:",
            f"📚 Uploaded Documents: {len(uploaded_docs)}",
            "\nCustom Documents:" if uploaded_docs else "\nNo custom documents uploaded."
        ]
        
        # List uploaded documents if any
        for doc in uploaded_docs:
            status_message.append(f"📄 {doc}")
            
        status_message.extend([
            "\n🔍 Using:",
            "- Custom documents (if uploaded)",
            "- Default NASP documents",
            "\nUse /reset to clear your session and remove uploaded documents."
        ])
        
        await update.message.reply_text("\n".join(status_message))

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle document uploads"""
        user_id = update.effective_user.id
        document = update.message.document
        
        # Initialize user session if doesn't exist
        if user_id not in user_sessions:
            user_sessions[user_id] = set()
        
        # Add document to session
        user_sessions[user_id].add(document.file_name)
        
        response = (
            f"📄 Received document: {document.file_name}\n"
            f"Current session has {len(user_sessions[user_id])} documents.\n"
            "Processing... I'll use this document for answering your questions.\n\n"
            "Use /status to see all uploaded documents or /reset to start fresh."
        )
        await update.message.reply_text(response)

    def run(self):
        """Run the bot"""
        self.application.run_polling()

def init_components() -> Tuple[Any, Any, Any]:
    """Initialize bot components"""
    try:
        load_dotenv()
        logger.info("=== Starting Bot Initialization ===")
        
        # Initialize LLM
        llm = ChatOpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=os.getenv("TOGETHER_API_KEY"),
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        )
        logger.info("LLM initialized successfully")
        
        # Initialize vector search
        vector_search = VectorSearch()
        vector_store = vector_search.load_or_create_vector_store()
        
        if not vector_store:
            raise Exception("Failed to initialize vector store")
            
        # Create retriever with similarity search
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 4  # Number of documents to retrieve
            }
        )
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant for the National Agency of Social Protection.
            Use the following pieces of context to answer the question. 
            If you don't know the answer, just say that you don't know.
            Keep your answers concise and relevant.
            
            Context: {context}"""),
            ("human", "{input}")
        ])
        
        # Create document chain
        document_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt,
            document_variable_name="context"
        )
        
        # Create retrieval chain
        qa_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=document_chain
        )
        
        logger.info("=== Bot Initialization Complete ===")
        return vector_store, vector_search, qa_chain
        
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}", exc_info=True)
        raise

def format_sources(sources: List[Document]) -> str:
    """Format source documents into a readable string with metadata"""
    unique_sources = {}
    
    for doc in sources:
        filename = os.path.basename(doc.metadata.get('source', 'Unknown'))
        page = doc.metadata.get('page', 'N/A')
        score = doc.metadata.get('score', 'N/A')
        
        if filename not in unique_sources:
            unique_sources[filename] = {
                'pages': set([page]),
                'score': score if isinstance(score, float) else 'N/A',
                'snippets': [doc.page_content[:200] + "..."]  # First 200 chars
            }
        else:
            unique_sources[filename]['pages'].add(page)
            if len(unique_sources[filename]['snippets']) < 2:  # Limit snippets
                unique_sources[filename]['snippets'].append(doc.page_content[:200] + "...")

    # Format into telegram-friendly markdown
    formatted_sources = ["📚 *Sources Used:*\n"]
    for filename, info in unique_sources.items():
        source_text = [
            f"📄 *Source*: `{filename}`",
            f"📑 *Pages*: {', '.join(map(str, sorted(info['pages'])))}",
        ]
        if info['score'] != 'N/A':
            source_text.append(f"🎯 *Relevance*: {info['score']:.2%}")
        
        source_text.append("\n🔍 *Relevant Excerpts*:")
        for i, snippet in enumerate(info['snippets'], 1):
            # Escape special characters for Telegram markdown
            safe_snippet = snippet.replace('_', '\\_').replace('*', '\\*').replace('`', '\\`')
            source_text.append(f"  {i}. {safe_snippet}")
        
        formatted_sources.append("\n".join(source_text))
    
    return "\n\n" + "\n\n---\n\n".join(formatted_sources)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming messages"""
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
            # Send the main answer
            await context.bot.send_message(
                chat_id=chat_id,
                text=response["answer"],
                parse_mode="Markdown"
            )
            
            # If there are sources, format and send them
            if "context" in response and response["context"]:
                sources_text = format_sources(response["context"])
                # Split long messages if needed (Telegram has 4096 char limit)
                if len(sources_text) > 4000:
                    chunks = [sources_text[i:i+4000] for i in range(0, len(sources_text), 4000)]
                    for chunk in chunks:
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=chunk,
                            parse_mode="Markdown"
                        )
                else:
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
        logger.error(f"Fatal error starting bot: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
