# tests/test_prediction_score.py
import unittest
from fastapi.testclient import TestClient
from datetime import datetime, timezone
import sqlite3
import os

from app import app, DB_PATH, init_db

class TestPredictionScore(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

        # Reset DB
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        init_db()

        # Add test user once here
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", ("alice", "pass123"))
            conn.commit()

    def test_valid_score(self):
        uid = "score-test-123"
        now = datetime.now(timezone.utc)

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO prediction_sessions (uid, username, timestamp, original_image, predicted_image)
                VALUES (?, ?, ?, ?, ?)
            """, (uid, "alice", now.isoformat(), "uploads/original/fake.jpg", "uploads/predicted/fake.jpg"))
            conn.execute("""
                INSERT INTO detection_objects (prediction_uid, label, score, box)
                VALUES (?, ?, ?, ?)
            """, (uid, "dog", 0.85, "[0, 0, 100, 100]"))
            conn.commit()  # commit to persist inserts

        response = self.client.get("/predictions/score/0.75", auth=("alice", "pass123"))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        print("Response JSON:", data)
        self.assertTrue(any("score" in obj for obj in data), "Expected at least one result with a score key")


    def test_invalid_score_below_range(self):
        response = self.client.get("/predictions/score/-0.5", auth=("alice", "pass123"))
        self.assertEqual(response.status_code, 400)

    def test_invalid_score_above_range(self):
        response = self.client.get("/predictions/score/1.5", auth=("alice", "pass123"))
        self.assertEqual(response.status_code, 400)
