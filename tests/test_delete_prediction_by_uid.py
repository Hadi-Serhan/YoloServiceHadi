# tests/test_delete_prediction_by_uid.py
import unittest
from fastapi.testclient import TestClient
from datetime import datetime, timezone
import os
import sqlite3

from app import app, DB_PATH, init_db

class TestDeletePredictionByUID(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

        # Reset the DB
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        init_db()

        # Create dummy image files
        self.uid = "test-delete-uid"
        self.original_path = f"uploads/original/{self.uid}.jpg"
        self.predicted_path = f"uploads/predicted/{self.uid}.jpg"

        os.makedirs("uploads/original", exist_ok=True)
        os.makedirs("uploads/predicted", exist_ok=True)

        with open(self.original_path, "wb") as f:
            f.write(b"original image content")
        with open(self.predicted_path, "wb") as f:
            f.write(b"predicted image content")

        # Add test user
        self.username = "alice"
        self.password = "pass123"
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (self.username, self.password))

        # Insert prediction associated with the user
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO prediction_sessions (uid, timestamp, original_image, predicted_image, username)
                VALUES (?, ?, ?, ?, ?)
            """, (
                self.uid,
                datetime.now(timezone.utc).isoformat(),
                self.original_path,
                self.predicted_path,
                self.username
            ))
            conn.execute("""
                INSERT INTO detection_objects (prediction_uid, label, score, box)
                VALUES (?, ?, ?, ?)
            """, (
                self.uid,
                "dog",
                0.9,
                "[0, 0, 100, 100]"
            ))

    def test_delete_existing_prediction(self):
        """Should delete prediction from DB and delete files"""
        response = self.client.delete(f"/prediction/{self.uid}", auth=(self.username, self.password))
        self.assertEqual(response.status_code, 200)
        self.assertIn("deleted successfully", response.json()["detail"])

        # Check DB is clean
        with sqlite3.connect(DB_PATH) as conn:
            session = conn.execute("SELECT * FROM prediction_sessions WHERE uid = ?", (self.uid,)).fetchone()
            self.assertIsNone(session)
            obj = conn.execute("SELECT * FROM detection_objects WHERE prediction_uid = ?", (self.uid,)).fetchone()
            self.assertIsNone(obj)

        # Check files deleted
        self.assertFalse(os.path.exists(self.original_path))
        self.assertFalse(os.path.exists(self.predicted_path))

    def test_delete_nonexistent_prediction(self):
        """Should return 404 if prediction UID doesn't exist"""
        response = self.client.delete("/prediction/nonexistent-uid", auth=(self.username, self.password))
        self.assertEqual(response.status_code, 404)
        self.assertIn("Prediction not found", response.json()["detail"])

    def test_delete_when_file_delete_raises_exception(self):
        """Should still delete DB record even if file deletion fails"""
        # Make the file read-only to trigger deletion failure
        os.chmod(self.original_path, 0o400)  # read-only
        os.chmod(self.predicted_path, 0o400)

        # Monkey-patch os.remove to raise an exception for coverage
        original_remove = os.remove

        def failing_remove(path):
            raise PermissionError(f"Cannot delete {path}")

        os.remove = failing_remove

        try:
            response = self.client.delete(f"/prediction/{self.uid}", auth=(self.username, self.password))
            self.assertEqual(response.status_code, 200)
            self.assertIn("deleted successfully", response.json()["detail"])
        finally:
            # Restore os.remove and permissions
            os.remove = original_remove
            os.chmod(self.original_path, 0o600)
            os.chmod(self.predicted_path, 0o600)
            if os.path.exists(self.original_path):
                os.remove(self.original_path)
            if os.path.exists(self.predicted_path):
                os.remove(self.predicted_path)

