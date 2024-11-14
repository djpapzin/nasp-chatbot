import unittest
import os
from dotenv import load_dotenv
from src.translation import AzureTranslator

# Load environment variables (if not already loaded)
load_dotenv()

class TestTranslation(unittest.TestCase):
    def setUp(self):
        """Setup method to create an instance of the translator before each test."""
        self.translator = AzureTranslator()

    def test_basic_translation(self):
        uzbek_text = "Salom dunyo!"
        expected_english = "Hello world!"  # Or a close approximation
        translated_text = self.translator.translate_uz_to_en(uzbek_text)
        self.assertIsNotNone(translated_text, "Translation result should not be None")
        self.assertIn("Hello", translated_text.lower(), "Translation should contain 'Hello'") # More flexible assertion
        self.assertIn("world", translated_text.lower(), "Translation should contain 'world'") # More flexible assertion

    def test_complex_sentence(self):
        uzbek_text = "Men bugun ishga bordim."
        translated_text = self.translator.translate_uz_to_en(uzbek_text)
        self.assertIsNotNone(translated_text, "Translation result should not be None")
        # Add assertions based on expected keywords or phrases in the translation

    def test_empty_input(self):
        uzbek_text = ""
        translated_text = self.translator.translate_uz_to_en(uzbek_text)
        self.assertEqual(translated_text, "", "Empty input should return an empty string") # Or assert None if that's your design

    def test_numbers_and_symbols(self):
        uzbek_text = "12345!@#$%"
        expected_english = "12345!@#$%"
        translated_text = self.translator.translate_uz_to_en(uzbek_text)
        self.assertEqual(translated_text, expected_english, "Numbers and symbols should remain unchanged")

    def test_special_characters(self):
        uzbek_text = "Oʻzbekiston"
        expected_english = "Uzbekistan" # Or a close approximation
        translated_text = self.translator.translate_uz_to_en(uzbek_text)
        self.assertIsNotNone(translated_text, "Translation result should not be None")
        self.assertIn("Uzbekistan", translated_text, "Translation should contain 'Uzbekistan'") # More flexible assertion

if __name__ == '__main__':
    unittest.main() 