import unittest
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient
from app import app
import os

class TestGetPrediction(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.username = "alice"
        self.password = "pass123"

        # Ensure test image exists for image tests
        self.test_image_path = "uploads/predicted/test.jpg"
        os.makedirs("uploads/predicted", exist_ok=True)
        with open(self.test_image_path, "wb") as f:
            f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 1024) 

    def tearDown(self):
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)

    @patch("services.prediction_uid_service.query_prediction_by_uid")
    def test_get_prediction_by_uid(self, mock_query):
        uid = "abc-123"
        mock_query.return_value = Mock(
            uid=uid,
            username=self.username,
            timestamp="2025-07-30T10:00:00Z",
            original_image="uploads/original/test.jpg",
            predicted_image=self.test_image_path
        )

        response = self.client.get(f"/prediction/{uid}", auth=(self.username, self.password))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["uid"], uid)

    @patch("queries.query_prediction_by_uid")
    def test_get_prediction_by_non_existing_uid(self, mock_query):
        mock_query.return_value = None
        response = self.client.get("/prediction/unknown-uid", auth=(self.username, self.password))
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Prediction not found")


    @patch("queries.get_prediction_image_path")
    def test_cannot_access_others_prediction(self, mock_get_path):
        mock_get_path.return_value = None 

        response = self.client.get(
            "/prediction/test-uid/image",
            headers={"Accept": "image/jpeg"},
            auth=(self.username, self.password)
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Prediction not found")
