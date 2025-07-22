# tests/test_get_prediction_controller.py

import unittest
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

from app import app
from db import get_db


# Fake class to simulate a SQLAlchemy DB model object
class FakePrediction:
    def __init__(self, uid, timestamp, original_image, predicted_image):
        self.uid = uid
        self.timestamp = timestamp
        self.original_image = original_image
        self.predicted_image = predicted_image


class TestGetPredictionByUID(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        def override_get_db():
            return Mock()
            
        # Override the get_db dependency
        app.dependency_overrides[get_db] = override_get_db

    def tearDown(self):
        # Clean up dependency overrides after each test
        app.dependency_overrides = {}

    @patch("app.query_prediction_by_uid")
    def test_prediction_found(self, mock_query):
        mock_query.return_value = FakePrediction(
            uid="123",
            timestamp="2023-01-01T12:00:00",
            original_image="input.png",
            predicted_image="output.png"
        )

        response = self.client.get("/prediction/123")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "uid": "123",
            "timestamp": "2023-01-01T12:00:00",
            "original_image": "input.png",
            "predicted_image": "output.png"
        })

        mock_query.assert_called_once()
        mock_query.assert_called_with(self.mock_db, "123")

    @patch("app.query_prediction_by_uid")
    def test_prediction_not_found(self, mock_query):
        mock_query.return_value = None
        response = self.client.get("/prediction/notfound")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Prediction not found"})