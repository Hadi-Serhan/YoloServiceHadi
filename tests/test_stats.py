import unittest
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient
from app import app

class TestStatsEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.username = "alice"
        self.password = "pass123"

    @patch("services.stats_service.get_detection_objects_for_recent_predictions")
    @patch("services.stats_service.count_recent_predictions")
    def test_stats_empty(self, mock_count, mock_objects):
        mock_count.return_value = 0
        mock_objects.return_value = []

        response = self.client.get("/stats", auth=(self.username, self.password))
        self.assertEqual(response.status_code, 200)
        stats = response.json()
        self.assertEqual(stats["total_predictions"], 0)
        self.assertEqual(stats["average_confidence_score"], 0.0)
        self.assertEqual(stats["most_common_labels"], {})

    @patch("services.stats_service.get_detection_objects_for_recent_predictions")
    @patch("services.stats_service.count_recent_predictions")
    def test_stats_with_recent_data(self, mock_count, mock_objects):
        mock_count.return_value = 1

        # Simulate 3 detections: person (2), dog (1)
        mock_objects.return_value = [
            Mock(score=0.9, label="person"),
            Mock(score=0.8, label="dog"),
            Mock(score=1.0, label="person"),
        ]

        response = self.client.get("/stats", auth=(self.username, self.password))
        self.assertEqual(response.status_code, 200)
        stats = response.json()

        self.assertEqual(stats["total_predictions"], 1)
        self.assertAlmostEqual(stats["average_confidence_score"], (0.9 + 0.8 + 1.0) / 3, places=4)
        self.assertEqual(stats["most_common_labels"], {"person": 2, "dog": 1})
