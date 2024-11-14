from typing import Optional
from azure.ai.translation.text import TextTranslator
from src.config import TRANSLATION_CONFIG

def translate_text(text: str, target_language: str, source_language: Optional[str] = None) -> str:
    """
    Translates text using Azure Translation Service.

    Args:
        text: The text to be translated.
        target_language: The target language code (e.g., "uz").
        source_language: The source language code (optional). If not provided, it will be auto-detected.

    Returns:
        The translated text.
    """
    try:
        translator = TextTranslator(
            endpoint=f"https://{TRANSLATION_CONFIG['azure_speech_region']}.api.cognitive.microsoft.com",
            key=TRANSLATION_CONFIG['azure_speech_key']
        )
        translated_text = translator.translate_text(
            text=text,
            to=target_language,
            from_=source_language
        )
        return translated_text.translations[0].text
    except Exception as e:
        print(f"Error during translation: {str(e)}")
        return text  # Return original text if translation fails 