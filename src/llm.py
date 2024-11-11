from together import Together
import streamlit as st

class LLMHandler:
    def __init__(self):
        """Initialize Together API client"""
        self.client = Together()
        self.model = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    def create_document_summary(self, docs):
        """Create a summary of available documents"""
        doc_summary = {}
        for doc in docs:
            filename = doc.metadata.get('filename', 'N/A')
            if filename not in doc_summary:
                doc_summary[filename] = {
                    'total_pages': doc.metadata.get('total_pages', 'N/A'),
                    'pages_referenced': set([doc.metadata.get('page', 1)])
                }
            else:
                doc_summary[filename]['pages_referenced'].add(doc.metadata.get('page', 1))

        # Format document summary
        docs_context = "Available documents:\n"
        for filename, info in doc_summary.items():
            docs_context += f"- {filename} (Total pages: {info['total_pages']})\n"
        
        return docs_context

    def generate_response(self, prompt, docs_and_scores):
        """Generate response using Together API with scored documents"""
        try:
            # Create document summary with scores
            docs_context = "Available documents:\n"
            for doc, score in docs_and_scores:
                filename = doc.metadata.get('filename', 'N/A')
                page = doc.metadata.get('page', 'N/A')
                docs_context += f"- {filename} (Page: {page}, Relevance: {score:.2f})\n"
            
            # Create context from documents
            context = f"{docs_context}\n\nContent:\n"
            for doc, score in docs_and_scores:
                context += f"\nFrom {doc.metadata.get('filename', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')}):\n"
                context += doc.page_content + "\n"
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
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
