# tests/test_auth.py
import io
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app import app
from db import get_db


class FakeUser:
    def __init__(self, username, password):
        self.username = username
        self.password = password


def generate_image_bytes():
    from PIL import Image
    img = Image.new("RGB", (100, 100), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


class TestAuthWithMocks(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.image_bytes = generate_image_bytes()

        # Replace FastAPI's get_db dependency with a mock
        self.mock_db = MagicMock()
        app.dependency_overrides[get_db] = lambda: self.mock_db

    def tearDown(self):
        app.dependency_overrides = {}

    @patch("services.predict_service.get_user")
    def test_predict_with_existing_user(self, mock_get_user):
        # Return a real object, not a mock
        mock_get_user.return_value = type("User", (), {"username": "user1", "password": "pass1"})()

        response = self.client.post(
            "/predict",
            files={"file": ("x.jpg", self.image_bytes, "image/jpeg")},
            auth=("user1", "pass1")
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction_uid", response.json())
        mock_get_user.assert_called_once()

    @patch("services.predict_service.get_user")
    def test_predict_invalid_password(self, mock_get_user):
        mock_get_user.return_value = type("User", (), {"username": "user1", "password": "correctpass"})()

        response = self.client.post(
            "/predict",
            files={"file": ("x.jpg", self.image_bytes, "image/jpeg")},
            auth=("user1", "wrongpass")
        )

        self.assertEqual(response.status_code, 401)
        self.assertIn("Invalid credentials", response.json()["detail"])


    @patch("services.predict_service.create_user")
    @patch("services.predict_service.get_user")
    def test_autoregister_new_user(self, mock_get_user, mock_create_user):
        mock_get_user.return_value = None  # Simulate no user

        response = self.client.post(
            "/predict",
            files={"file": ("x.jpg", self.image_bytes, "image/jpeg")},
            auth=("newuser", "newpass")
        )

        self.assertEqual(response.status_code, 200)
        mock_create_user.assert_called_once()


    @patch("queries.get_user")
    def test_authentication_required(self, mock_get_user):
        response = self.client.get("/predictions/count")
        self.assertEqual(response.status_code, 401)
