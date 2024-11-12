import os
import logging
from pathlib import Path
from typing import Tuple, Any, List

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

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send welcome message when /start is issued"""
    welcome_message = """
    👋 Welcome to the NASP Chatbot!
    
    I can help you find information about social protection programs and policies.
    You can:
    1. Ask me questions about social protection documents
    2. Upload PDF, DOCX, or TXT files for me to analyze
    
    Just send me your question or upload a document to get started!
    """
    print(f"\nUser {update.message.from_user.username} started the bot")
    await update.message.reply_text(welcome_message)

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

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle document uploads"""
    try:
        doc = update.message.document
        print(f"\nReceived document from {update.message.from_user.username}: {doc.file_name}")
        
        # Check file type
        allowed_types = ['application/pdf', 'application/msword', 
                        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                        'text/plain']
        
        if doc.mime_type not in allowed_types:
            error_msg = "❌ Sorry, I can only process PDF, DOCX, or TXT files."
            print(f"\nRejected document: {error_msg}")
            await update.message.reply_text(error_msg)
            return
        
        # Download file
        processing_msg = "📥 Processing your document..."
        print(f"\nBot: {processing_msg}")
        await update.message.reply_text(processing_msg)
        file = await context.bot.get_file(doc.file_id)
        file_path = f"temp_{doc.file_name}"
        await file.download_to_drive(file_path)
        
        try:
            # Process document using DocumentManager
            doc_manager = DocumentManager()
            success, result = doc_manager.process_file(file_path, doc.file_name)
            
            if success:
                # Add to vector store
                vector_store.add_documents(result)
                vector_store.save_local("faiss_index", "default_index")
                success_msg = (f"✅ Successfully processed and indexed: {doc.file_name}\n\n"
                             "You can now ask questions about this document!")
                print(f"\nBot: {success_msg}")
                await update.message.reply_text(success_msg)
            else:
                error_msg = f"❌ Failed to process {doc.file_name}: {result}"
                print(f"\nBot: {error_msg}")
                await update.message.reply_text(error_msg)
                
        finally:
            # Clean up temp file
            if os.path.exists(file_path):
                os.remove(file_path)
            
    except Exception as e:
        logger.error(f"Error handling document: {str(e)}", exc_info=True)
        error_msg = "Sorry, I encountered an error processing your document. Please try again."
        print(f"\nBot: {error_msg}")
        await update.message.reply_text(error_msg)

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
