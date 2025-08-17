# tests/test_get_image.py

import unittest
from unittest.mock import patch, mock_open
from fastapi.testclient import TestClient
from app import app


class TestImageRetrieval(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.username = "alice"
        self.password = "pass123"

    @patch("services.image_service.user_owns_image", return_value=True)
    @patch("services.image_service.os.path.exists", return_value=True)
    def test_invalid_type_returns_400(self, mock_exists, mock_owns):
        response = self.client.get("/image/invalid/somefile.jpg", auth=(self.username, self.password))
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid image type", response.text)

    @patch("services.image_service.user_owns_image", return_value=True)
    @patch("services.image_service.os.path.exists", return_value=False)
    def test_file_does_not_exist_returns_404(self, mock_exists, mock_owns):
        response = self.client.get("/image/original/nonexistent.jpg", auth=(self.username, self.password))
        self.assertEqual(response.status_code, 404)
        self.assertIn("Image not found", response.text)

    @patch("services.image_service.user_owns_image", return_value=False)
    @patch("services.image_service.os.path.exists", return_value=True)
    def test_file_exists_but_not_owned_returns_404(self, mock_exists, mock_owns):
        response = self.client.get("/image/original/testimage.jpg", auth=(self.username, self.password))
        self.assertEqual(response.status_code, 404)
        self.assertIn("Access denied", response.text)
        

    @patch("services.image_service.user_owns_image", return_value=True)
    @patch("services.image_service.os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open)
    def test_successful_image_access(self, mock_file, mock_exists, mock_owns):
        mock_file.return_value.read.return_value = b"real content"

        response = self.client.get("/image/original/test_success.jpg", auth=(self.username, self.password))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"real content")

