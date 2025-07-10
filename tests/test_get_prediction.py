# test_get_prediction.py

import io
import os
from PIL import Image
import unittest
from fastapi.testclient import TestClient
from app import app, init_db, DB_PATH


class TestGetPrediction(unittest.TestCase):

    def setUp(self):  
       self.client = TestClient(app)     

       # Remove existing database 
       if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
       
       # Initialize a clean database    
       init_db()
       
       # Create a simple test image
       self.test_image = Image.new('RGB', (100, 100), color='red')
       self.image_bytes = io.BytesIO()
       self.test_image.save(self.image_bytes, format='JPEG')
       self.image_bytes.seek(0)
       
       response = self.client.post(
               "/predict",
               files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
           )
       self.expectedData = response.json()
       

    def test_get_prediction_by_uid(self):
        response = self.client.get(f"/prediction/{self.expectedData['prediction_uid']}")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data['uid'], self.expectedData['prediction_uid'])

    def test_get_prediction_by_non_existed_uid(self):
        response = self.client.get(f"/prediction/12344")
        self.assertEqual(response.status_code, 404)