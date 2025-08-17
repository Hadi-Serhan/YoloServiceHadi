import os
import shutil
import uuid
import time
from PIL import Image
from ultralytics import YOLO
import secrets

from queries import (
    get_user, create_user,
    save_prediction_session,
    save_detection_object,
)

UPLOAD_DIR = "uploads/original"
PREDICTED_DIR = "uploads/predicted"
model = YOLO("yolov8n.pt")  # Load model once

def process_prediction(file, db, username=None, password=None):
    if username:
        user = get_user(db, username)
        if user is None:
            create_user(db, username, password)
        elif not secrets.compare_digest(user.password, password):
            raise ValueError("Invalid credentials")

    ext = os.path.splitext(file.filename)[1]
    uid = str(uuid.uuid4())
    original_path = os.path.join(UPLOAD_DIR, uid + ext)
    predicted_path = os.path.join(PREDICTED_DIR, uid + ext)

    with open(original_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    start_time = time.time()

    results = model(original_path, device="cpu")
    annotated_frame = results[0].plot()
    Image.fromarray(annotated_frame).save(predicted_path)

    save_prediction_session(db, uid, original_path, predicted_path, username)

    labels = []
    for box in results[0].boxes:
        label_idx = int(box.cls[0].item())
        label = model.names[label_idx]
        score = float(box.conf[0])
        bbox = str(box.xyxy[0].tolist())
        save_detection_object(db, uid, label, score, bbox)
        labels.append(label)

    return {
        "prediction_uid": uid,
        "detection_count": len(results[0].boxes),
        "labels": labels,
        "time_took": round(time.time() - start_time, 2)
    }
