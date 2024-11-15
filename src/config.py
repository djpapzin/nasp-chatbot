CHATBOT_CONFIG = {
    "name": "NASP Chatbot",
    "welcome_message": """
Welcome! This is a prototype chatbot for the National Agency of Social Protection. 
You can use it to ask questions about a library of reports, evaluations, research, and other documents.

Hello. Please enter your question in the chat box to get started.
""",
    "available_languages": {
        "en": "English",
        "ru": "Русский",
        "uz": "O'zbek"
    }
}

LANGUAGE_CONFIG = {
    "default_language": "en",
    "available_languages": {
        "en": "English",
        "ru": "Русский",
        "uz": "O'zbek"
    },
    "needs_translation": ["ru", "uz"]
}

EMBEDDINGS_CONFIG = {
    "model": "text-embedding-ada-002",
    "persist_directory": "faiss_index",
    "chunk_size": 1000,
    "chunk_overlap": 200
}

PROMPT_CONFIG = {
    "en": {
        "system_prompt": """You are an expert researcher that can answer questions about social protection in Uzbekistan. 
Use the following pieces of retrieved context to answer the question. Use clear and professional language, 
and organize the summary in a logical manner using appropriate formatting such as headings, subheadings, and bullet points. 
If the information needed to answer the question is not available in the context then say that you don't know.

Context: {context}
"""
    }
} 