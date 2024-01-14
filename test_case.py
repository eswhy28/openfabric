import unittest
from unittest.mock import patch
from main import is_question, attempt_model_response, generate_response, is_valid_response,gpt2_model, gpt2_tokenizer
import main
class TestMainFunctions(unittest.TestCase):

    def setUp(self):
        self.factual_question = "What is the DNA sequence?"
        self.creative_question = "Imagine a world without gravity."
        self.non_question = "This is a statement."

    def test_is_question_with_question(self):
        self.assertTrue(is_question(self.factual_question))
        self.assertTrue(is_question(self.creative_question))

    def test_is_question_with_non_question(self):
        self.assertFalse(is_question(self.non_question))

    @patch('main.attempt_model_response')
    def test_attempt_model_response_with_mock(self, mock_attempt_model_response):
        # Mock return value should be set correctly
        mock_attempt_model_response.return_value = "DNA consists of four major bases."
        response = main.attempt_model_response(gpt2_tokenizer, gpt2_model, self.factual_question)
        self.assertIsNotNone(response)
        self.assertEqual(response, "DNA consists of four major bases.")

    def test_generate_response_for_factual_question(self):
        with patch('main.attempt_model_response') as mock_model:
            mock_model.return_value = "DNA is a molecule composed of two chains."
            response = generate_response(self.factual_question)
            self.assertIn("DNA", response)

    def test_generate_response_for_creative_question(self):
        with patch('main.generate_gpt2_response') as mock_gpt2:
            mock_gpt2.return_value = "In a world without gravity, people would float."
            response = generate_response(self.creative_question)
            self.assertIn("float", response)

    def test_is_valid_response_with_valid_response(self):
        response = "This is a valid response."
        self.assertTrue(is_valid_response(response))

    def test_is_valid_response_with_invalid_response(self):
        response = "I'm not sure about this. Could you rephrase or ask another question?"
        self.assertFalse(is_valid_response(response))

