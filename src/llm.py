from together import Together
import streamlit as st
import os
from langchain_openai import ChatOpenAI

class LLMHandler:
    def __init__(self):
        """Initialize Together API client and LangChain LLM"""
        self.client = Together()
        self.model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        
        # Initialize LangChain LLM
        self.llm = ChatOpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=os.getenv("TOGETHER_API_KEY"),
            model=self.model_name,
            temperature=0.7,
            max_tokens=1000
        )

    def create_document_context(self, docs_and_scores):
        """Create a formatted context of available documents"""
        unique_docs = {}
        
        for doc, score in docs_and_scores:
            # Extract just the filename without path and clean it up
            raw_filename = doc.metadata.get('source', 'Unknown Document')
            filename = os.path.basename(raw_filename)  # Remove path
            filename = filename.replace('_', ' ')  # Replace underscores with spaces
            filename = os.path.splitext(filename)[0]  # Remove file extension
            
            if filename not in unique_docs:
                unique_docs[filename] = {
                    'score': score,
                    'pages': set([doc.metadata.get('page', 1)]),
                    'total_pages': doc.metadata.get('total_pages', 'Unknown')
                }
            else:
                unique_docs[filename]['pages'].add(doc.metadata.get('page', 1))
        
        # Format document list
        context = "I have access to the following documents:\n\n"
        for filename, info in unique_docs.items():
            context += f"- {filename}\n"
        
        return context

    def generate_response(self, prompt, docs_and_scores):
        """Generate response using LLM"""
        if not docs_and_scores:
            return "I'm sorry, I couldn't find any relevant documents to answer your question."
            
        # Special handling for document listing request
        if "list" in prompt.lower() and "documents" in prompt.lower():
            return self.create_document_context(docs_and_scores)
            
        try:
            # Create context from documents
            docs_context = "Documents used:\n"
            for doc, score in docs_and_scores:
                filename = doc.metadata.get('filename', 'N/A')
                page = doc.metadata.get('page', 'N/A')
                docs_context += f"- {filename} (Page: {page}, Relevance: {score:.2f})\n"
            
            # Create context from documents
            context = f"{docs_context}\n\nContent:\n"
            for doc, score in docs_and_scores:
                context += f"\nFrom {doc.metadata.get('filename', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')}):\n"
                context += doc.page_content + "\n"
            
            # Generate response using LangChain LLM
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": """You are a helpful assistant that answers questions based on the provided context. 
                    When discussing documents, always mention their filenames and page counts when available.
                    Include relevance scores when citing information to show confidence in the sources.
                    If you cannot find specific information in the context, say so."""},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            error_message = f"Error generating response: {str(e)}"
            st.error(error_message)
            print(f"Error: {error_message}")
            return None

    def deduplicate_results(self, docs_and_scores, similarity_threshold=0.95):
        """Deduplicate similar results"""
        unique_results = []
        seen_content = set()
        
        for doc, score in docs_and_scores:
            content_hash = hash(doc.page_content[:100])  # Use first 100 chars as signature
            if content_hash not in seen_content:
                unique_results.append((doc, score))
                seen_content.add(content_hash)
        
        return unique_results
