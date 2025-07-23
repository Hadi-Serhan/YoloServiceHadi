# tests/test_auth.py
import io
import os
import sqlite3
import unittest
from fastapi.testclient import TestClient
from app import app, init_db, DB_PATH

USERNAME = "user1"
PASSWORD = "pass1"


def load_image_bytes(filename):
    path = os.path.join(os.getcwd(), filename)
    with open(path, "rb") as f:
        return io.BytesIO(f.read())


class AuthTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Reinitialize clean DB
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        init_db()
        # Add test user
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (USERNAME, PASSWORD))

        cls.image_bytes = load_image_bytes("beatles.jpeg")

    def test_status_no_auth(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok"})

    def test_predict_with_auth(self):
        r = self.client.post(
            "/predict",
            files={"file": ("beatles.jpeg", self.image_bytes, "image/jpeg")},
            auth=(USERNAME, PASSWORD)
        )
        self.assertEqual(r.status_code, 200)
        json_data = r.json()
        self.assertIn("prediction_uid", json_data)
        uid = json_data["prediction_uid"]

        # Session should be retrievable by same user
        r2 = self.client.get(f"/prediction/{uid}", auth=(USERNAME, PASSWORD))
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r2.json()["uid"], uid)

    def test_predict_without_auth(self):
        r = self.client.post(
            "/predict",
            files={"file": ("pic1.jpg", load_image_bytes("pic1.jpg"), "image/jpeg")}
        )
        self.assertEqual(r.status_code, 200)
        json_data = r.json()
        self.assertIn("prediction_uid", json_data)
        uid = json_data["prediction_uid"]

        # Prediction retrieval should be forbidden
        r2 = self.client.get(f"/prediction/{uid}")
        self.assertEqual(r2.status_code, 401)

    def test_protected_endpoints_require_auth(self):
        r = self.client.get("/predictions/count")
        self.assertEqual(r.status_code, 401)
        r = self.client.get("/labels")
        self.assertEqual(r.status_code, 401)

    def test_access_with_wrong_password(self):
        # First, register correctly
        self.client.post("/predict", files={"file": ("x.jpg", self.image_bytes, "image/jpeg")},
                         auth=("alice", "pass123"))

        # Then try with wrong password
        response = self.client.get("/predictions/count", auth=("alice", "wrongpass"))
        self.assertEqual(response.status_code, 401)

    def test_access_with_nonexistent_user(self):
        response = self.client.get("/predictions/count", auth=("ghost", "nopass"))
        self.assertEqual(response.status_code, 401)

    def test_autoregistration_on_first_use(self):
        # First request with a new user – should succeed
        response = self.client.post(
            "/predict",
            files={"file": ("x.jpg", self.image_bytes, "image/jpeg")},
            auth=("bobnew", "newpass")
        )
        self.assertEqual(response.status_code, 200)

        # Now manually insert the same username to simulate conflict
        with sqlite3.connect(DB_PATH) as conn:
            try:
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", ("bobfail", "conflict"))
                conn.commit()
            except sqlite3.IntegrityError:
                pass  # If already exists, skip

        # Trigger second request that will try to auto-register "bobfail" and fail
        response2 = self.client.post(
            "/predict",
            files={"file": ("x.jpg", self.image_bytes, "image/jpeg")},
            auth=("bobfail", "wrongpass")  # intentionally wrong pass to force DB insert
        )
        self.assertEqual(response2.status_code, 401)
        self.assertIn("Invalid credentials", response2.json()["detail"])

