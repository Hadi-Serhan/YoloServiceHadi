import time
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.responses import FileResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from ultralytics import YOLO
from PIL import Image
import sqlite3
import os
import uuid
import shutil
from datetime import datetime, timedelta
from collections import Counter
import base64
import secrets
from typing import Optional

# Disable GPU usage
import torch
torch.cuda.is_available = lambda: False

app = FastAPI()
security = HTTPBasic()

UPLOAD_DIR = "uploads/original"
PREDICTED_DIR = "uploads/predicted"
DB_PATH = "predictions.db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)

# Download the AI model (tiny model ~6MB)
model = YOLO("yolov8n.pt")  

# Initialize SQLite
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        # Create the predictions main table to store the prediction session
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_sessions (
                uid TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                original_image TEXT,
                predicted_image TEXT,
                username TEXT,
                FOREIGN KEY (username) REFERENCES users(username)
            )
        """)
        
        # Create the objects table to store individual detected objects in a given image
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detection_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_uid TEXT,
                label TEXT,
                score REAL,
                box TEXT,
                FOREIGN KEY (prediction_uid) REFERENCES prediction_sessions (uid)
            )
        """)
        # Create the users table for authentication
        conn.execute("""
                     CREATE TABLE If NOT EXISTS users (
                        username TEXT PRIMARY KEY,
                        password TEXT NOT NULL
                     )
                     """)
        
        # Create index for faster queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prediction_uid ON detection_objects (prediction_uid)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_label ON detection_objects (label)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_score ON detection_objects (score)")

init_db()

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Verifies the username/password against the users table.
    Returns the username if valid and raises 401 otherwise
    """

    correct_username = credentials.username
    correct_password = credentials.password
    
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT password FROM users WHERE username = ?", (correct_username,)).fetchone()
        if not row or not secrets.compare_digest(row[0], correct_password):
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Basic"},
            )
    return correct_username


def save_prediction_session(uid, original_image, predicted_image, username):
    """
    Save prediction session to database
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO prediction_sessions (uid, original_image, predicted_image, username)
            VALUES (?, ?, ?, ?)
        """, (uid, original_image, predicted_image, username))

def save_detection_object(prediction_uid, label, score, box):
    """
    Save detection object to database
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO detection_objects (prediction_uid, label, score, box)
            VALUES (?, ?, ?, ?)
        """, (prediction_uid, label, score, str(box)))

@app.post("/predict")
def predict(file: UploadFile = File(...), credentials: Optional[HTTPBasicCredentials] = Depends(HTTPBasic(auto_error=False))):
    """
    Predict objects in an image
    """
    
    username = None
    if credentials:
        get_current_username(credentials)
        username = credentials.username
        
    start_time = time.time()
    
    ext = os.path.splitext(file.filename)[1]
    uid = str(uuid.uuid4())
    original_path = os.path.join(UPLOAD_DIR, uid + ext)
    predicted_path = os.path.join(PREDICTED_DIR, uid + ext)

    with open(original_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    results = model(original_path, device="cpu")

    annotated_frame = results[0].plot()  # NumPy image with boxes
    annotated_image = Image.fromarray(annotated_frame)
    annotated_image.save(predicted_path)

    save_prediction_session(uid, original_path, predicted_path, username)
    
    detected_labels = []
    for box in results[0].boxes:
        label_idx = int(box.cls[0].item())
        label = model.names[label_idx]
        score = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        save_detection_object(uid, label, score, bbox)
        detected_labels.append(label)

    processing_time = round(time.time() - start_time, 2)

    return {
        "prediction_uid": uid, 
        "detection_count": len(results[0].boxes),
        "labels": detected_labels,
        "time_took": processing_time,
    }

@app.get("/prediction/{uid}")
def get_prediction_by_uid(uid: str, username: str = Depends(get_current_username)):
    """
    Get prediction session by uid with all detected objects
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        # Get prediction session
        session = conn.execute("SELECT * FROM prediction_sessions WHERE uid = ?", (uid,)).fetchone()
        if not session:
            raise HTTPException(status_code=404, detail="Prediction not found")
        # Check if the session belongs to the current user
        if session['username'] != username:
            raise HTTPException(status_code=403, detail="Access denied")
        # Get all detection objects for this prediction
        objects = conn.execute(
            "SELECT * FROM detection_objects WHERE prediction_uid = ?", 
            (uid,)
        ).fetchall()
        
        return {
            "uid": session["uid"],
            "timestamp": session["timestamp"],
            "original_image": session["original_image"],
            "predicted_image": session["predicted_image"],
            "detection_objects": [
                {
                    "id": obj["id"],
                    "label": obj["label"],
                    "score": obj["score"],
                    "box": obj["box"]
                } for obj in objects
            ]
        }

@app.get("/predictions/label/{label}")
def get_predictions_by_label(label: str, username: str = Depends(get_current_username)):
    """
    Get prediction sessions containing objects with specified label
    """
    
    if label not in model.names.values():
        raise HTTPException(status_code=404, detail="Label not supported")
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT DISTINCT ps.uid, ps.timestamp
            FROM prediction_sessions ps
            JOIN detection_objects do ON ps.uid = do.prediction_uid
            WHERE do.label = ? AND ps.username = ?
        """, (label, username)).fetchall()
        
        return [{"uid": row["uid"], "timestamp": row["timestamp"]} for row in rows]

@app.get("/predictions/score/{min_score}")
def get_predictions_by_score(min_score: float, username: str = Depends(get_current_username)):
    """
    Get prediction sessions containing objects with score >= min_score
    """
    if not (min_score >= 0 and min_score <= 1):
        raise HTTPException(status_code=400, detail="Score must be between 0 and 1")
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT DISTINCT ps.uid, ps.timestamp
            FROM prediction_sessions ps
            JOIN detection_objects do ON ps.uid = do.prediction_uid
            WHERE do.score >= ? AND username = ?
        """, (min_score, username)).fetchall()
        
        return [{"uid": row["uid"], "timestamp": row["timestamp"]} for row in rows]

@app.get("/image/{type}/{filename}")
def get_image(type: str, filename: str, username: str = Depends(get_current_username)):
    """
    Get image by type and filename
    """
    if type not in ["original", "predicted"]:
        raise HTTPException(status_code=400, detail="Invalid image type")
    path = os.path.join("uploads", type, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)

@app.get("/predictions/count")
def get_prediction_counter(username: str = Depends(get_current_username)):
    """
    Get total number of predictions within the past week
    """
    one_week_ago = datetime.now() - timedelta(days=7)
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("""
                           SELECT COUNT(*) as count 
                           FROM prediction_sessions 
                           WHERE timestamp >= ? AND username = ?
                           """,(one_week_ago.isoformat(), username)).fetchone()
    return {"count": row["count"]}
    
@app.get("/labels")
def get_recent_labels(username: str = Depends(get_current_username)):
    """
    Get all unique labels detected in the last 7 days.
    """
    one_week_ago = datetime.now() - timedelta(days=7)

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT DISTINCT label
            FROM detection_objects
            WHERE prediction_uid IN (
                SELECT uid FROM prediction_sessions
                WHERE timestamp >= ? AND ps.username = ?
            )
        """, (one_week_ago.isoformat(), username)).fetchall()

    return {"labels": [row["label"] for row in rows]}

    
@app.get("/prediction/{uid}/image")
def get_prediction_image(uid: str, request: Request, username: str = Depends(get_current_username)):
    """
    Get prediction image by uid
    """
    accept = request.headers.get("accept", "")
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("""
                           SELECT predicted_image 
                           FROM prediction_sessions 
                           WHERE uid = ? AND username = ?
                           """, (uid, username)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Prediction not found")
        image_path = row[0]

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Predicted image file not found")

    if "image/png" in accept:
        return FileResponse(image_path, media_type="image/png")
    elif "image/jpeg" in accept or "image/jpg" in accept:
        return FileResponse(image_path, media_type="image/jpeg")
    else:
        # If the client doesn't accept image, respond with 406 Not Acceptable
        raise HTTPException(status_code=406, detail="Client does not accept an image format")


@app.delete("/prediction/{uid}")
def delete_prediction(uid: str, username: str = Depends(get_current_username)):
    """
    Delete a specific prediction and its related image files and detection objects.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        # Fetch the prediction row
        prediction = conn.execute("""
                                  SELECT * FROM prediction_sessions WHERE uid = ? AND username = ?
                                  """, (uid, username)).fetchone()
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")

        # Delete related detection objects
        conn.execute("""DELETE FROM detection_objects
                     WHERE prediction_uid = ? AND username = ?
                     """, (uid, username))
        
        # Delete the prediction session
        conn.execute("DELETE FROM prediction_sessions WHERE uid = ? AND username", (uid, username))

    # Delete files from disk (fail silently if file missing)
    for path_key in ["original_image", "predicted_image"]:
        file_path = prediction[path_key]
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Warning: Failed to delete {file_path} — {e}")

    return {"detail": f"Prediction {uid} deleted successfully."}

@app.get("/stats")
def get_stats(username: str = Depends(get_current_username)):
    """
    Get overall statistics and analytics about predictions in the last 7 days.
    """
    one_week_ago = datetime.now() - timedelta(days=7)

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        # Get total number of predictions
        total_predictions_row = conn.execute("""
            SELECT COUNT(*) as count
            FROM prediction_sessions
            WHERE timestamp >= ? AND username = ?
        """, (one_week_ago.isoformat(), username)).fetchone()
        total_predictions = total_predictions_row["count"]

        # Get all detection objects for those predictions
        rows = conn.execute("""
            SELECT label, score
            FROM detection_objects
            WHERE prediction_uid IN (
                SELECT uid FROM prediction_sessions
                WHERE timestamp >= ? AND username = ?
            )
        """, (one_week_ago.isoformat(), username)).fetchall()

    scores = [row["score"] for row in rows]
    labels = [row["label"] for row in rows]

    # Compute average confidence score
    avg_confidence = round(sum(scores) / len(scores), 4) if scores else 0.0

    # Count most common labels
    label_counts = Counter(labels)

    return {
        "total_predictions": total_predictions,
        "average_confidence_score": avg_confidence,
        "most_common_labels": dict(label_counts)
    }
    

@app.get("/health")
def health():
    """
    Health check endpoint
    """
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
