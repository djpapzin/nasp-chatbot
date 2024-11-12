import streamlit as st
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from typing import List
import os

logger = logging.getLogger(__name__)

# Define the system prompt
SYSTEM_PROMPT = """Your task is to be an expert researcher that can answer questions.  Use the following pieces of retrieved context to answer the question.  Use clear and professional language, and organize the summary in a logical manner using appropriate formatting such as headings, subheadings, and bullet points.  If the information needed to answer the question is not available in the context then say that you don't know.

Context: {context}
"""

# Create the chat prompt template
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}")
])

class UI:
    @staticmethod
    def setup_page():
        """Configure initial page settings"""
        st.set_page_config(
            page_title="NASP Chatbot",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    @staticmethod
    def show_header():
        """Display header and welcome message"""
        st.title("NASP Chatbot")
        st.markdown("""
            Welcome! This is a prototype chatbot for the National Agency of Social Protection. 
            You can use it to ask questions about a library of reports, evaluations, research, and other documents.
        """)

    @staticmethod
    def show_file_uploader(doc_manager, vector_store) -> None:
        """Display file upload interface in sidebar"""
        with st.sidebar:
            st.header("📄 Upload Additional Documents")
            uploaded_files = st.file_uploader(
                "Upload PDF, DOCX, or TXT files",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                help="Maximum file size: 200MB"
            )
            
            if uploaded_files:
                for file in uploaded_files:
                    with st.spinner(f'Processing {file.name}...'):
                        try:
                            success, result = doc_manager.process_file(file, file.name)
                            if success and vector_store is not None:
                                vector_store.add_documents(result)
                                st.success(f'Successfully processed: {file.name}')
                            else:
                                st.error(f'Error processing {file.name}: {result}')
                        except Exception as e:
                            st.error(f'Error processing {file.name}: {str(e)}')

    @staticmethod
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
                    'snippets': [doc.page_content[:200] + "..."]  # First 200 chars of content
                }
            else:
                unique_sources[filename]['pages'].add(page)
                if len(unique_sources[filename]['snippets']) < 2:  # Limit to 2 snippets per source
                    unique_sources[filename]['snippets'].append(doc.page_content[:200] + "...")

        # Format into markdown
        formatted_sources = []
        for filename, info in unique_sources.items():
            source_text = f"📄 **Source**: {filename}\n"
            source_text += f"📑 **Pages**: {', '.join(map(str, sorted(info['pages'])))}\n"
            if info['score'] != 'N/A':
                source_text += f"🎯 **Relevance**: {info['score']:.2%}\n"
            source_text += "\n🔍 **Relevant Excerpts**:\n"
            for i, snippet in enumerate(info['snippets'], 1):
                source_text += f"  {i}. {snippet}\n"
            formatted_sources.append(source_text)
            
        return "\n---\n".join(formatted_sources)

    @staticmethod
    def show_chat_interface(vector_store, llm_handler):
        """Display chat interface with RAG"""
        if vector_store is None:
            st.error("Vector store not initialized. Please check your credentials and try again.")
            return
        
        # Create the RAG chain
        try:
            retriever = vector_store.as_retriever()
            question_answer_chain = create_stuff_documents_chain(
                llm_handler.llm,
                QA_PROMPT
            )
            qa_chain = create_retrieval_chain(retriever, question_answer_chain)

            # Create container for chat interface
            chat_container = st.container()
            
            with chat_container:
                # Initialize chat history if it doesn't exist
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                    # Add initial greeting with simpler message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Hello. Please enter your question in the chat box to get started."
                    })
                
                # Display all messages in the chat history
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
                
                # Chat input
                if prompt := st.chat_input("What would you like to know?"):
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    
                    # Display user message
                    with st.chat_message("user"):
                        st.write(prompt)
                    
                    # Generate and display assistant response
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            try:
                                response = qa_chain.invoke({
                                    "input": prompt
                                })
                                
                                if response and "answer" in response:
                                    st.write(response["answer"])
                                    # Add assistant response to chat history
                                    st.session_state.messages.append({
                                        "role": "assistant", 
                                        "content": response["answer"]
                                    })
                                    
                                    # Show sources if available
                                    if "context" in response:
                                        with st.expander("📚 View Sources", expanded=False):
                                            formatted_sources = UI.format_sources(response["context"])
                                            st.markdown(formatted_sources)
                                else:
                                    st.error("I couldn't find relevant information to answer your question.")
                                            
                            except Exception as e:
                                st.error("I encountered an error while generating a response. Please try again.")
                                logger.error(f"Error generating response: {str(e)}", exc_info=True)
        except Exception as e:
            st.error(f"Error initializing chat interface: {str(e)}")
            return
