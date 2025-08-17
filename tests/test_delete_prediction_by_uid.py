# tests/test_delete_prediction_by_uid.py

import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app import app


class FakePrediction:
    def __init__(self, uid, original_image, predicted_image, username):
        self.uid = uid
        self.original_image = original_image
        self.predicted_image = predicted_image
        self.username = username
        self.timestamp = "2023-01-01T00:00:00"


class TestDeletePredictionByUID(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.username = "alice"
        self.password = "pass123"
        self.uid = "test-delete-uid"
        self.original_path = f"uploads/original/{self.uid}.jpg"
        self.predicted_path = f"uploads/predicted/{self.uid}.jpg"

    @patch("services.delete_service.get_prediction_by_uid_and_user")
    @patch("services.delete_service.delete_detection_objects_by_uid")
    @patch("services.delete_service.delete_prediction_session")
    @patch("services.delete_service.os.remove")
    @patch("services.delete_service.os.path.exists", return_value=True)
    def test_delete_existing_prediction(
        self, mock_exists, mock_remove, mock_delete_session, mock_delete_objects, mock_get_prediction
    ):
        mock_get_prediction.return_value = FakePrediction(
            uid=self.uid,
            original_image=self.original_path,
            predicted_image=self.predicted_path,
            username=self.username,
        )

        response = self.client.delete(f"/prediction/{self.uid}", auth=(self.username, self.password))

        self.assertEqual(response.status_code, 200)
        self.assertIn("deleted successfully", response.json()["detail"])
        mock_remove.assert_any_call(self.original_path)
        mock_remove.assert_any_call(self.predicted_path)
        mock_delete_session.assert_called_once()
        mock_delete_objects.assert_called_once()

    @patch("services.delete_service.get_prediction_by_uid_and_user", return_value=None)
    def test_delete_nonexistent_prediction(self, mock_get_prediction):
        response = self.client.delete(f"/prediction/does-not-exist", auth=(self.username, self.password))
        self.assertEqual(response.status_code, 404)
        self.assertIn("Prediction not found", response.json()["detail"])

    @patch("services.delete_service.get_prediction_by_uid_and_user")
    @patch("services.delete_service.delete_detection_objects_by_uid")
    @patch("services.delete_service.delete_prediction_session")
    @patch("services.delete_service.os.remove")
    @patch("services.delete_service.os.path.exists", return_value=True)
    def test_delete_when_file_remove_raises_exception(
        self, mock_exists, mock_remove, mock_delete_session, mock_delete_objects, mock_get_prediction
    ):
        mock_get_prediction.return_value = FakePrediction(
            uid=self.uid,
            original_image=self.original_path,
            predicted_image=self.predicted_path,
            username=self.username,
        )
        mock_remove.side_effect = PermissionError("file delete failed")

        response = self.client.delete(f"/prediction/{self.uid}", auth=(self.username, self.password))
        self.assertEqual(response.status_code, 200)
        self.assertIn("deleted successfully", response.json()["detail"])
