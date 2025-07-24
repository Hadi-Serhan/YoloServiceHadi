# tests/test_get_prediction.py
import io
import os
from PIL import Image
import unittest
from fastapi.testclient import TestClient
import sqlite3

from app import app, init_db, DB_PATH

class TestGetPrediction(unittest.TestCase):

    def setUp(self):  
        self.client = TestClient(app)     

        # Remove existing database 
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        
        # Initialize a clean database    
        init_db()
        
        # Create test user
        self.username = "alice"
        self.password = "pass123"
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (self.username, self.password))

        # Create a simple test image
        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.image_bytes = io.BytesIO()
        self.test_image.save(self.image_bytes, format='JPEG')
        self.image_bytes.seek(0)

        # Make a prediction
        response = self.client.post(
            "/predict",
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")},
            auth=(self.username, self.password)
        )
        self.assertEqual(response.status_code, 200)
        self.expectedData = response.json()

        # Save predicted image path for deletion test
        self.predicted_image_path = os.path.join("uploads", "predicted", f"{self.expectedData['prediction_uid']}.jpg")


    def test_get_prediction_by_uid(self):
        response = self.client.get(
            f"/prediction/{self.expectedData['prediction_uid']}",
            auth=(self.username, self.password)
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['uid'], self.expectedData['prediction_uid'])

    def test_get_prediction_by_non_existed_uid(self):
        response = self.client.get(
            "/prediction/12344",
            auth=(self.username, self.password)
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Prediction not found") 

    def test_cannot_access_others_prediction(self):
        self.image_bytes.seek(0)
        self.client.post("/predict", files={"file": ("x.jpg", self.image_bytes, "image/jpeg")},
                        auth=("alice", "pass123"))

        uid = self.expectedData['prediction_uid']
        self.image_bytes.seek(0)
        self.client.post("/predict", files={"file": ("dummy.jpg", self.image_bytes, "image/jpeg")},
                auth=("bob", "passbob"))
        response = self.client.get(f"/prediction/{uid}", auth=("bob", "passbob"))
        self.assertEqual(response.status_code, 403)

    def test_unacceptable_accept_header(self):
        self.image_bytes.seek(0)
        response = self.client.post("/predict", files={"file": ("img.jpg", self.image_bytes, "image/jpeg")},
                                    auth=("alice", "pass123"))
        uid = response.json()["prediction_uid"]

        response = self.client.get(f"/prediction/{uid}/image",
                                headers={"Accept": "application/json"},
                                auth=("alice", "pass123"))
        self.assertEqual(response.status_code, 406)
        self.assertEqual(response.json()["detail"], "Client does not accept an image format")

    def test_prediction_image_edge_cases(self):
        uid = self.expectedData["prediction_uid"]

        response = self.client.get(f"/prediction/{uid}/image",
                                headers={"Accept": "application/json"},
                                auth=(self.username, self.password))
        self.assertEqual(response.status_code, 406)

        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("SELECT predicted_image FROM prediction_sessions WHERE uid = ?", (uid,)).fetchone()
            image_path = row[0]

        if os.path.exists(image_path):
            os.remove(image_path)

        response = self.client.get(f"/prediction/{uid}/image",
                                headers={"Accept": "image/jpeg"},
                                auth=(self.username, self.password))
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Predicted image file not found")

    def test_png_accept_header_success(self):
        self.image_bytes.seek(0)
        r = self.client.post("/predict", files={"file": ("img.png", self.image_bytes, "image/png")},
                             auth=(self.username, self.password))
        uid = r.json()["prediction_uid"]

        response = self.client.get(
            f"/prediction/{uid}/image",
            headers={"Accept": "image/png"},
            auth=(self.username, self.password)
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")  

    def test_jpeg_accept_header_success(self):
        uid = self.expectedData["prediction_uid"]
        response = self.client.get(
            f"/prediction/{uid}/image",
            headers={"Accept": "image/jpeg"},
            auth=(self.username, self.password)
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg") 


    def test_prediction_image_uid_not_found(self):
        response = self.client.get(
            "/prediction/fake-uid/image",
            headers={"Accept": "image/jpeg"},
            auth=(self.username, self.password)
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Prediction not found")
