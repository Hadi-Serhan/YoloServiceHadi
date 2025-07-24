# tests/test_stats.py
import unittest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta, timezone
import sqlite3
import os

from app import app, DB_PATH, init_db

class TestStatsEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

        # Reset database
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        init_db()

        self.now = datetime.now(timezone.utc)
        self.recent_uid = "recent-uid"
        self.old_uid = "old-uid"

        self.username = "alice"
        self.password = "pass123"

        # Insert test user
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (self.username, self.password))

    def insert_prediction(self, uid: str, timestamp: datetime):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO prediction_sessions (uid, username, timestamp, original_image, predicted_image)
                VALUES (?, ?, ?, ?, ?)
            """, (uid, self.username, timestamp.isoformat(), "original.jpg", "predicted.jpg"))

    def insert_object(self, prediction_uid: str, label: str, score: float):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO detection_objects (prediction_uid, label, score, box)
                VALUES (?, ?, ?, ?)
            """, (prediction_uid, label, score, "[0, 0, 50, 50]"))

    def test_stats_empty(self):
        """Should return 0s and empty dict when no predictions"""
        response = self.client.get("/stats", auth=(self.username, self.password))
        self.assertEqual(response.status_code, 200)
        stats = response.json()
        self.assertEqual(stats["total_predictions"], 0)
        self.assertEqual(stats["average_confidence_score"], 0.0)
        self.assertEqual(stats["most_common_labels"], {})

    def test_stats_with_recent_data(self):
        """Should return stats for recent predictions only"""
        # Insert recent prediction with 3 objects
        self.insert_prediction(self.recent_uid, self.now - timedelta(days=1))
        self.insert_object(self.recent_uid, "person", 0.9)
        self.insert_object(self.recent_uid, "dog", 0.8)
        self.insert_object(self.recent_uid, "person", 1.0)

        # Insert old prediction (should be ignored)
        self.insert_prediction(self.old_uid, self.now - timedelta(days=10))
        self.insert_object(self.old_uid, "car", 0.5)

        response = self.client.get("/stats", auth=(self.username, self.password))
        self.assertEqual(response.status_code, 200)
        stats = response.json()

        self.assertEqual(stats["total_predictions"], 1)
        self.assertAlmostEqual(stats["average_confidence_score"], (0.9 + 0.8 + 1.0) / 3, places=4)
        self.assertEqual(stats["most_common_labels"], {"person": 2, "dog": 1})
