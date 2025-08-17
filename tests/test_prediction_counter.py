import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app import app

class TestPredictionCounter(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.username = "alice"
        self.password = "pass123"

    @patch("services.count_service.count_predictions_in_last_week")
    def test_prediction_counter_empty(self, mock_count):
        mock_count.return_value = 0
        response = self.client.get("/predictions/count", auth=(self.username, self.password))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["count"], 0)

    @patch("services.count_service.count_predictions_in_last_week")
    def test_prediction_counter_last_week(self, mock_count):
        mock_count.return_value = 3
        response = self.client.get("/predictions/count", auth=(self.username, self.password))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["count"], 3)
