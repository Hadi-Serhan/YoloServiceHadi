import io
import os
import sqlite3
import unittest
from fastapi.testclient import TestClient
from app import app, init_db, DB_PATH

client = TestClient(app)

USERNAME = "user1"
PASSWORD = "pass1"

def load_image_bytes(filename):
    path = os.path.join(os.getcwd(), filename)
    with open(path, "rb") as f:
        return io.BytesIO(f.read())

class AuthTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reinitialize clean DB
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        init_db()
        # Add test user
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (USERNAME, PASSWORD))

    def test_status_no_auth(self):
        r = client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok"})

    def test_predict_with_auth(self):
        img_file = load_image_bytes("beatles.jpeg")
        r = client.post(
            "/predict",
            files={"file": ("beatles.jpeg", img_file, "image/jpeg")},
            auth=(USERNAME, PASSWORD)
        )
        self.assertEqual(r.status_code, 200)
        json_data = r.json()
        self.assertIn("prediction_uid", json_data)
        uid = json_data["prediction_uid"]

        # Session should be retrievable by same user
        r2 = client.get(f"/prediction/{uid}", auth=(USERNAME, PASSWORD))
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r2.json()["uid"], uid)

    def test_predict_without_auth(self):
        img_file = load_image_bytes("pic1.jpg")
        r = client.post(
            "/predict",
            files={"file": ("pic1.jpg", img_file, "image/jpeg")}
        )
        self.assertEqual(r.status_code, 200)
        json_data = r.json()
        self.assertIn("prediction_uid", json_data)
        uid = json_data["prediction_uid"]

        # Prediction retrieval should be forbidden
        r2 = client.get(f"/prediction/{uid}")
        self.assertEqual(r2.status_code, 401)

    def test_protected_endpoints_require_auth(self):
        r = client.get("/predictions/count")
        self.assertEqual(r.status_code, 401)
        r = client.get("/labels")
        self.assertEqual(r.status_code, 401)
