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
    "needs_translation": ["ru", "uz"]  # Languages that need translation from English
} 