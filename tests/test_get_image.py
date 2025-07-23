# tests/test_get_image.py
import unittest
from fastapi.testclient import TestClient
from datetime import datetime, timezone
import sqlite3
import os

from app import app, DB_PATH, init_db

class TestImageRetrieval(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

        # Reset DB
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        init_db()

        # Create test user
        self.username = "alice"
        self.password = "pass123"
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (self.username, self.password))

    def test_invalid_type_returns_400(self):
        response = self.client.get("/image/invalid/somefile.jpg", auth=(self.username, self.password))
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid image type", response.text)

    def test_file_does_not_exist_returns_404(self):
        response = self.client.get("/image/original/nonexistent.jpg", auth=(self.username, self.password))
        self.assertEqual(response.status_code, 404)
        self.assertIn("Image not found", response.text)

    def test_file_exists_but_not_owned_returns_404(self):
        # Create dummy image file
        dummy_path = "uploads/original/testimage.jpg"
        os.makedirs(os.path.dirname(dummy_path), exist_ok=True)
        with open(dummy_path, "wb") as f:
            f.write(b"fake image content")

        # Add another user (bob) who "owns" the file
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", ("bob", "bobpass"))
            conn.execute("""
                INSERT INTO prediction_sessions (uid, username, timestamp, original_image, predicted_image)
                VALUES (?, ?, ?, ?, ?)
            """, ("img123", "bob", datetime.now(timezone.utc).isoformat(), dummy_path, "uploads/predicted/other.jpg"))

        # Alice tries to access Bob's image
        response = self.client.get("/image/original/testimage.jpg", auth=(self.username, self.password))
        self.assertEqual(response.status_code, 404)
        self.assertIn("Access denied", response.text)

        # Clean up
        os.remove(dummy_path)


    def test_successful_image_access(self):
        # Create dummy original image
        image_path = "uploads/original/test_success.jpg"
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(b"real content")

        # Add to DB for this user
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO prediction_sessions (uid, username, timestamp, original_image, predicted_image)
                VALUES (?, ?, datetime('now'), ?, ?)
            """, ("success123", self.username, image_path, "uploads/predicted/ignored.jpg"))

        # Make request
        response = self.client.get("/image/original/test_success.jpg", auth=(self.username, self.password))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"real content")
