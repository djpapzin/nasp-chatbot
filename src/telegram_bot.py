import os
import logging
from pathlib import Path
from typing import Tuple, Any, List

# Telegram imports
from telegram import Update
from telegram.ext import (
    Application, 
    CommandHandler, 
    MessageHandler, 
    filters, 
    ContextTypes
)

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_together import TogetherEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

# Local imports
from document_manager import DocumentManager

# Configure logging
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
    """Initialize bot components"""
    try:
        load_dotenv()
        logger.info("=== Starting Bot Initialization ===")
        
        # Initialize embeddings and document manager
        embeddings = TogetherEmbeddings(
            model="BAAI/bge-large-en-v1.5",
            together_api_key=os.getenv("TOGETHER_API_KEY")
        )
        doc_manager = DocumentManager()
        
        # Load vector store with default documents
        logger.info("Loading FAISS vector store...")
        try:
            vector_store = FAISS.load_local(
                folder_path="faiss_index",
                embeddings=embeddings,
                index_name="default_index"
            )
            logger.info("Existing vector store loaded successfully")
        except Exception as e:
            logger.warning(f"No existing vector store found, creating new one with default documents")
            # Load default documents
            default_docs = doc_manager.load_default_documents()
            if default_docs:
                vector_store = FAISS.from_documents(default_docs, embeddings)
                vector_store.save_local("faiss_index", "default_index")
                logger.info("Created new vector store with default documents")
            else:
                # Fallback to empty store if no default docs
                vector_store = FAISS.from_texts(["Initial document"], embeddings)
                vector_store.save_local("faiss_index", "default_index")
                logger.warning("Created empty vector store")

        # Initialize LLM
        llm = ChatOpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=os.getenv("TOGETHER_API_KEY"),
            model="mistralai/Mixtral-8x7B-Instruct-v0.1"
        )
        
        # Create retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
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
        
        # Create document chain first
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
        return vector_store, llm, qa_chain, DocumentManager()
        
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}", exc_info=True)
        raise

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send welcome message when /start is issued."""
    welcome_message = """
    Welcome to the NASP Chatbot! 👋
    
    I can help you find information about social protection programs and policies.
    Just ask me a question and I'll do my best to help!
    """
    await update.message.reply_text(welcome_message)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming messages"""
    try:
        # Get response from RAG chain
        response = qa_chain.invoke({
            "input": update.message.text
        })
        
        # Send response
        await update.message.reply_text(response["answer"])
        
        # Optionally show sources
        if "context" in response and response["context"]:
            sources_text = "\n\nSources:\n"
            for doc in response["context"]:
                source = doc.metadata.get('source', 'Unknown')
                sources_text += f"- {source}\n"
            await update.message.reply_text(sources_text)
            
    except Exception as e:
        logger.error(f"Error handling message: {str(e)}", exc_info=True)
        await update.message.reply_text(
            "I apologize, but I encountered an error processing your request."
        )

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle document uploads"""
    try:
        # Get document
        doc = update.message.document
        
        # Check file type
        allowed_types = ['application/pdf', 'application/msword', 
                        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                        'text/plain']
        
        if doc.mime_type not in allowed_types:
            await update.message.reply_text(
                "❌ Sorry, I can only process PDF, DOCX, or TXT files."
            )
            return
        
        # Download file
        file = await context.bot.get_file(doc.file_id)
        file_path = f"temp_{doc.file_name}"
        await file.download_to_drive(file_path)
        
        try:
            # Process document
            success, result = doc_manager.process_file(file_path, doc.file_name)
            
            if success:
                # Add to vector store
                vector_store.add_documents(result)
                vector_store.save_local("faiss_index", "default_index")
                await update.message.reply_text(
                    f"✅ Successfully processed and indexed: {doc.file_name}"
                )
            else:
                await update.message.reply_text(
                    f"❌ Failed to process {doc.file_name}: {result}"
                )
                
        finally:
            # Clean up temp file
            if os.path.exists(file_path):
                os.remove(file_path)
            
    except Exception as e:
        logger.error(f"Error handling document: {str(e)}", exc_info=True)
        await update.message.reply_text(
            "Sorry, I encountered an error processing your document."
        )

def main() -> None:
    """Start the bot"""
    try:
        # Initialize components
        global vector_store, llm, qa_chain, doc_manager
        vector_store, llm, qa_chain, doc_manager = init_components()
        
        # Create application
        application = Application.builder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        application.add_handler(MessageHandler(filters.ATTACHMENT, handle_document))
        
        # Start bot
        logger.info("Starting bot...")
        application.run_polling()
        
    except Exception as e:
        logger.error(f"Fatal error starting bot: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
