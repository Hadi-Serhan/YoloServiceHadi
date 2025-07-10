import unittest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta, timezone
import sqlite3
import os

from app import app, DB_PATH, init_db

class TestUniqueLabels(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

        # Reset the database
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        init_db()

        # Helper: Insert a prediction session
    def insert_prediction(self, uid: str, timestamp: datetime):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO prediction_sessions (uid, timestamp, original_image, predicted_image)
                VALUES (?, ?, ?, ?)
            """, (uid, timestamp.isoformat(), "original.jpg", "predicted.jpg"))

    # Helper: Insert a detection object
    def insert_object(self, prediction_uid: str, label: str, score: float = 0.9, box="[0,0,50,50]"):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO detection_objects (prediction_uid, label, score, box)
                VALUES (?, ?, ?, ?)
            """, (prediction_uid, label, score, box))

    def test_no_labels(self):
        """Should return empty list if no labels in the last week"""
        response = self.client.get("/labels")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["labels"], [])

    def test_unique_labels_last_week(self):
        """Should return only unique labels from the last 7 days"""
        now = datetime.now(timezone.utc)

        # Insert recent prediction with 2 objects
        uid_recent = "recent-123"
        self.insert_prediction(uid_recent, now - timedelta(days=1))
        self.insert_object(uid_recent, "cat")
        self.insert_object(uid_recent, "dog")

        # Insert duplicate label
        self.insert_object(uid_recent, "cat")

        # Insert old prediction that should be ignored
        uid_old = "old-123"
        self.insert_prediction(uid_old, now - timedelta(days=10))
        self.insert_object(uid_old, "elephant")

        response = self.client.get("/labels")
        self.assertEqual(response.status_code, 200)

        labels = response.json()["labels"]
        self.assertIn("cat", labels)
        self.assertIn("dog", labels)
        self.assertNotIn("elephant", labels)
        self.assertEqual(sorted(labels), sorted(set(labels)))  # ensure uniqueness

