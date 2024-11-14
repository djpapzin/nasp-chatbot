from typing import Optional
import os
from azure.ai.translation.text import TextTranslationClient, TranslatorCredential
from azure.ai.translation.text.models import InputTextItem
import logging
from src.config import LANGUAGE_CONFIG

logger = logging.getLogger(__name__)

class Translator:
    def __init__(self):
        """Initialize the translator with Azure credentials"""
        try:
            self.key = os.getenv("AZURE_TRANSLATOR_KEY")
            self.endpoint = os.getenv("AZURE_TRANSLATOR_ENDPOINT")
            self.region = os.getenv("AZURE_TRANSLATOR_REGION")
            
            if not all([self.key, self.endpoint, self.region]):
                raise ValueError("Missing required Azure Translator credentials")
            
            self.credential = TranslatorCredential(self.key, self.region)
            self.client = TextTranslationClient(
                endpoint=self.endpoint,
                credential=self.credential
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize translator: {str(e)}")
            raise

    def translate(self, 
                 text: str, 
                 target_language: str, 
                 source_language: Optional[str] = None) -> str:
        """
        Translate text to target language
        
        Args:
            text: Text to translate
            target_language: Target language code (e.g., 'uz', 'ru', 'en')
            source_language: Source language code (optional)
            
        Returns:
            Translated text
        """
        try:
            # Skip translation if target language is the same as source
            if source_language and source_language == target_language:
                return text
                
            # Only translate if target language needs translation
            if target_language not in LANGUAGE_CONFIG["needs_translation"]:
                return text
            
            input_text_elements = [InputTextItem(text=text)]
            
            response = self.client.translate(
                content=input_text_elements,
                to=[target_language],
                from_parameter=source_language
            )
            
            # Get the translation for the first (and only) input
            translation = response[0] if response else None
            
            if translation and translation.translations:
                return translation.translations[0].text
            else:
                logger.warning(f"No translation returned for text: {text[:100]}...")
                return text
                
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            return text  # Return original text if translation fails
            
    def detect_language(self, text: str) -> str:
        """
        Detect the language of input text
        
        Args:
            text: Text to detect language for
            
        Returns:
            Detected language code
        """
        try:
            response = self.client.detect_language(
                content=[text],
                kind="None"
            )
            
            if response and response[0].primary_language:
                return response[0].primary_language.language
            else:
                logger.warning("Could not detect language, defaulting to English")
                return "en"
                
        except Exception as e:
            logger.error(f"Language detection failed: {str(e)}")
            return "en"  # Default to English on error

    def translate_response(self, 
                         response: str, 
                         target_language: str, 
                         preserve_formatting: bool = True) -> str:
        """
        Translate bot response while preserving markdown formatting
        
        Args:
            response: Bot response to translate
            target_language: Target language code
            preserve_formatting: Whether to preserve markdown formatting
            
        Returns:
            Translated response
        """
        try:
            if target_language not in LANGUAGE_CONFIG["needs_translation"]:
                return response
                
            if preserve_formatting:
                # Split on markdown elements while preserving them
                parts = []
                current_text = ""
                in_markdown = False
                
                for char in response:
                    if char in ['*', '_', '#', '`', '-']:
                        if current_text:
                            if not in_markdown:
                                # Translate accumulated text
                                current_text = self.translate(
                                    current_text, 
                                    target_language
                                )
                            parts.append(current_text)
                            current_text = ""
                        parts.append(char)
                        in_markdown = not in_markdown
                    else:
                        current_text += char
                
                # Handle any remaining text
                if current_text:
                    if not in_markdown:
                        current_text = self.translate(
                            current_text, 
                            target_language
                        )
                    parts.append(current_text)
                
                return "".join(parts)
            else:
                # Simple translation without preserving formatting
                return self.translate(response, target_language)
                
        except Exception as e:
            logger.error(f"Response translation failed: {str(e)}")
            return response
