import unittest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta, timezone
import sqlite3
import os

from app import app, DB_PATH, init_db

class TestPredictionCounter(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

        # Reset DB
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        init_db()

    def insert_prediction(self, timestamp: datetime):
        with sqlite3.connect(DB_PATH) as conn:
            uid = f"test-{timestamp.timestamp()}"
            conn.execute("""
                INSERT INTO prediction_sessions (uid, timestamp, original_image, predicted_image)
                VALUES (?, ?, ?, ?)
            """, (uid, timestamp.isoformat(), "original.jpg", "predicted.jpg"))

    def test_prediction_counter_empty(self):
        """Should return 0 when there are no predictions"""
        response = self.client.get("/prediction/counter")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["count"], 0)

    def test_prediction_counter_last_week(self):
        """Should count only predictions from last 7 days"""
        now = datetime.now(timezone.utc)

        # Insert 3 recent and 2 old predictions
        self.insert_prediction(now - timedelta(days=1))
        self.insert_prediction(now - timedelta(days=2))
        self.insert_prediction(now - timedelta(days=6))
        self.insert_prediction(now - timedelta(days=8))
        self.insert_prediction(now - timedelta(days=30))

        response = self.client.get("/prediction/counter")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["count"], 3)