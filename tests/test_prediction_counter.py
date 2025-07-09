import unittest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import sqlite3
import os

from app import app, DB_PATH  # adjust the import if your file isn't named main.py

client = TestClient(app)

class TestPredictionCountEndpoint(unittest.TestCase):

    def setUp(self):
        # Set up a clean test database
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        from app import init_db  # re-run the schema setup
        init_db()

    def insert_prediction(self, timestamp: datetime):
        with sqlite3.connect(DB_PATH) as conn:
            uid = f"test-{timestamp.timestamp()}"
            conn.execute("""
                INSERT INTO prediction_sessions (uid, timestamp, original_image, predicted_image)
                VALUES (?, ?, ?, ?)
            """, (uid, timestamp.isoformat(), "original.jpg", "predicted.jpg"))

    def test_prediction_count_last_week(self):
        now = datetime.utcnow()

        # Insert predictions: 3 within last 7 days, 2 older
        self.insert_prediction(now - timedelta(days=1))  # should count
        self.insert_prediction(now - timedelta(days=3))  # should count
        self.insert_prediction(now - timedelta(days=6))  # should count
        self.insert_prediction(now - timedelta(days=8))  # should not count
        self.insert_prediction(now - timedelta(days=30))  # should not count

        # Call the endpoint
        response = client.get("/prediction/count")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["count"], 3)

    def test_prediction_count_empty(self):
        # No data
        response = client.get("/prediction/count")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["count"], 0)

if __name__ == '__main__':
    unittest.main()
