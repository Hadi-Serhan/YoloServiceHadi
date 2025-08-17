import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app import app

class TestPredictionScore(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.username = "alice"
        self.password = "pass123"

    @patch("services.score_service.get_predictions_by_score")
    def test_valid_score(self, mock_query):
        mock_query.return_value = [
            ("uid123", "2025-07-31T12:00:00", 0.85),
            ("uid456", "2025-07-30T14:30:00", 0.90),
        ]
        response = self.client.get("/predictions/score/0.75", auth=(self.username, self.password))
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(isinstance(data, list))
        self.assertTrue(all("score" in d for d in data))

    def test_invalid_score_below_range(self):
        response = self.client.get("/predictions/score/-0.5", auth=(self.username, self.password))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Score must be between 0 and 1")

    def test_invalid_score_above_range(self):
        response = self.client.get("/predictions/score/1.5", auth=(self.username, self.password))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Score must be between 0 and 1")
