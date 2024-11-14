CHATBOT_CONFIG = {
    "name": "NASP Chatbot",
    "description": {
        "en": "Welcome! This is a prototype chatbot for the National Agency of Social Protection. "
              "You can use it to ask questions about a library of reports, evaluations, research and other documents.",
        "ru": "Добро пожаловать! Это прототип чат-бота Национального агентства социальной защиты. "
              "Вы можете задавать вопросы о библиотеке отчетов, оценок, исследований и других документов.",
        "uz": "Xush kelibsiz! Bu Ijtimoiy himoya milliy agentligining prototip chat-boti. "
              "Siz hisobotlar, baholashlar, tadqiqotlar va boshqa hujjatlar kutubxonasi haqida savollar berishingiz mumkin."
    },
    "opening_message": {
        "en": "Hello. Please enter your question in the chat box to get started.",
        "ru": "Здравствуйте. Пожалуйста, введите ваш вопрос в чат, чтобы начать.",
        "uz": "Salom. Boshlash uchun chat maydoniga savolingizni kiriting."
    },
    "documents": [
        "Analysis of the state system in Uzbekistan (Employment and welfare services for persons with disabilities)",
        "Unpaid Care and Domestic Work in Uzbekistan (ESCAP 2023)",
        "Social Protection Innovation and Learning (SPIL) in Uzbekistan",
        "Prioritising universal health insurance in Uzbekistan",
        "Exploring Pathways to Decent Employment (Informality Report)",
        "Road to Recovery Report",
    ]
}

# Prompt Templates for different languages
PROMPT_CONFIG = {
    "en": {
        "system_prompt": """Your task is to be an expert researcher that can answer questions.  
        Use the following pieces of retrieved context to answer the question.  
        Use clear and professional language, and organize the summary in a logical manner using appropriate formatting 
        such as headings, subheadings, and bullet points.  
        If the information needed to answer the question is not available in the context then say that you don't know.

        Context: {context}
        """,
    },
    "ru": {
        "system_prompt": """Ваша задача - быть экспертом-исследователем, который может отвечать на вопросы.
        Используйте следующие части полученного контекста для ответа на вопрос.
        Используйте четкий и профессиональный язык, организуйте резюме логическим образом, 
        используя соответствующее форматирование, такое как заголовки, подзаголовки и маркированные списки.
        Если информация, необходимая для ответа на вопрос, отсутствует в контексте, скажите, что вы не знаете.

        Контекст: {context}
        """,
    },
    "uz": {
        "system_prompt": """Sizning vazifangiz savollarga javob bera oladigan ekspert-tadqiqotchi bo'lishdir.
        Savolga javob berish uchun quyidagi kontekst qismlaridan foydalaning.
        Aniq va professional tildan foydalaning, xulosani sarlavhalar, kichik sarlavhalar va 
        belgilangan ro'yxatlar kabi tegishli formatlashdan foydalangan holda mantiqiy tarzda tashkil eting.
        Agar savolga javob berish uchun zarur ma'lumot kontekstda mavjud bo'lmasa, bilmasligingizni ayting.

        Kontekst: {context}
        """,
    }
}

# Azure Translation Service Config
TRANSLATION_CONFIG = {
    "azure_speech_key": "",  # To be loaded from environment variables
    "azure_speech_region": "",  # To be loaded from environment variables
    "translation_model": "mistralai/Mixtral-8x7B-Instruct-v0.1"
}

# Language configurations
LANGUAGE_CONFIG = {
    "available_languages": {
        "en": "English",
        "ru": "Русский",
        "uz": "O'zbek"
    },
    "default_language": "en",
    "needs_translation": ["uz"]  # Languages that need Azure translation service
} 