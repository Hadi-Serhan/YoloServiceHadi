import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app import app

class TestUniqueLabels(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.username = "alice"
        self.password = "pass123"

    @patch("services.label_service.get_recent_labels")
    def test_no_labels(self, mock_get_labels):
        mock_get_labels.return_value = []
        response = self.client.get("/labels", auth=(self.username, self.password))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["labels"], [])

    @patch("services.label_service.get_recent_labels")
    def test_unique_labels_last_week(self, mock_get_labels):
        mock_get_labels.return_value = ["cat", "dog", "cat", "elephant"]
        response = self.client.get("/labels", auth=(self.username, self.password))
        self.assertEqual(response.status_code, 200)

        labels = response.json()["labels"]
        self.assertIn("cat", labels)
        self.assertIn("dog", labels)
        self.assertIn("elephant", labels)

    @patch("services.label_service.get_predictions_by_label")
    @patch("services.label_service.model")
    def test_predictions_by_valid_label(self, mock_model, mock_get_preds):
        mock_model.names.values.return_value = {"dog", "cat", "elephant"}
        mock_get_preds.return_value = [
            ("uid123", "2025-07-31T12:00:00"),
            ("uid456", "2025-07-30T11:00:00")
        ]

        response = self.client.get("/predictions/label/dog", auth=(self.username, self.password))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(any("uid123" in d["uid"] for d in data))

    @patch("services.label_service.model")
    def test_invalid_label_gives_404(self, mock_model):
        mock_model.names.values.return_value = {"cat", "dog", "car"}

        response = self.client.get("/predictions/label/doesntexist", auth=(self.username, self.password))
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Label not supported")
