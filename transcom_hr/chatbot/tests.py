import os
from django.test import TestCase, Client
from django.urls import reverse
from django.conf import settings
from chatbot.services import get_vector_store, generate_retention_response

class ChatbotTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.url = reverse('chatbot_api')

    def test_policies_file_exists(self):
        """Verify the retention policies knowledge base file exists."""
        policies_path = os.path.join(settings.BASE_DIR, 'chatbot', 'data', 'transcom_retention_policies.txt')
        self.assertTrue(os.path.exists(policies_path), "Policies text file should exist.")
        
        with open(policies_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("SECTION 1: OVERTIME FATIGUE COUNTERMEASURES", content)
            self.assertIn("SECTION 2: COMMUTE & DISTANCE MITIGATION", content)
            self.assertIn("SECTION 3: LOW ENGAGEMENT & MANAGER FRICTION", content)

    def test_missing_api_key_response(self):
        """Verify generate_retention_response returns standard instruction when API key is unconfigured."""
        # Save original settings key
        original_key = settings.GEMINI_API_KEY
        
        try:
            # Set to default placeholder or empty to test defensive fallback
            settings.GEMINI_API_KEY = "your_gemini_api_key_here"
            response_placeholder = generate_retention_response("What is the overtime cap?")
            self.assertIn("System Notice: The GEMINI_API_KEY is currently not configured", response_placeholder)
            
            settings.GEMINI_API_KEY = ""
            response_empty = generate_retention_response("What is the overtime cap?")
            self.assertIn("System Notice: The GEMINI_API_KEY is currently not configured", response_empty)
            
        finally:
            settings.GEMINI_API_KEY = original_key

    def test_api_view_method_check(self):
        """Verify the API endpoint enforces POST request method."""
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 405)
        
    def test_api_view_empty_message(self):
        """Verify that empty request messages are rejected with bad request status."""
        response = self.client.post(self.url, data='{"message": ""}', content_type='application/json')
        self.assertEqual(response.status_code, 400)
        self.assertIn("Message cannot be empty", response.json().get('error'))
