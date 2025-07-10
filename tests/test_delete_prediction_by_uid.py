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

        # Insert into DB
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO prediction_sessions (uid, timestamp, original_image, predicted_image)
                VALUES (?, ?, ?, ?)
            """, (
                self.uid,
                datetime.now(timezone.utc).isoformat(),
                self.original_path,
                self.predicted_path
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
        response = self.client.delete(f"/prediction/{self.uid}")
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
        response = self.client.delete("/prediction/nonexistent-uid")
        self.assertEqual(response.status_code, 404)
        self.assertIn("Prediction not found", response.json()["detail"])

    def test_delete_when_files_already_missing(self):
        """Should still delete DB record even if files don't exist"""
        os.remove(self.original_path)
        os.remove(self.predicted_path)

        response = self.client.delete(f"/prediction/{self.uid}")
        self.assertEqual(response.status_code, 200)
        self.assertIn("deleted successfully", response.json()["detail"])
