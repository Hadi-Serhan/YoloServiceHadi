# services/predict_service.py
import os
import uuid
import time
from PIL import Image
from ultralytics import YOLO
import secrets
from services.validators import (
    validate_mime_and_ext,
    sniff_image_or_415,
    sanitize_filename,
)
from infra import enforce_db_quota

from queries import (
    get_user,
    create_user,
    save_prediction_session,
    save_detection_object,
)

UPLOAD_DIR = "uploads/original"
PREDICTED_DIR = "uploads/predicted"
model = YOLO("yolov8n.pt")  # Load model once

MAX_BYTES = 10 * 1024 * 1024  # 10 MB
CHUNK = 1 * 1024 * 1024  # 1 MB


def process_prediction(file, db, username=None, password=None):
    if username:
        enforce_db_quota(db, username, monthly_limit=100)  # e.g. 100/month
        user = get_user(db, username)
        if user is None:
            create_user(db, username, password)
        elif not secrets.compare_digest(user.password, password):
            raise ValueError("Invalid credentials")

    # --- Ensure directories exist ---
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(PREDICTED_DIR, exist_ok=True)

    # --- Validate MIME + extension (JPEG/PNG) ---
    validate_mime_and_ext(file)  # uses file.content_type and file.filename

    # --- Safe/normalized filename & extension ---
    safe_name = sanitize_filename(file.filename)
    _, ext = os.path.splitext(safe_name)
    uid = str(uuid.uuid4())
    original_path = os.path.join(UPLOAD_DIR, uid + ext.lower())
    predicted_path = os.path.join(PREDICTED_DIR, uid + ext.lower())

    # --- Stream upload to disk with a 10 MB cap ---
    total = 0
    try:
        with open(original_path, "wb") as out:
            while True:
                chunk = file.file.read(CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_BYTES:
                    raise ValueError("File too large (max 10MB)")
                out.write(chunk)
    except Exception:
        # Cleanup partial file on any failure (including size)
        try:
            if os.path.exists(original_path):
                os.remove(original_path)
        except OSError:
            pass
        # Raise appropriate HTTP-style error up the stack
        from fastapi import HTTPException

        raise HTTPException(status_code=413, detail="File too large (max 10MB)")

    # --- Sniff the saved file to ensure it's truly an image ---
    try:
        with open(original_path, "rb") as fh:
            sniff_image_or_415(fh.read(64 * 1024))
    except Exception:
        # Invalid/corrupted image -> cleanup and fail
        try:
            os.remove(original_path)
        except OSError:
            pass
        from fastapi import HTTPException

        raise HTTPException(status_code=415, detail="Invalid or corrupted image")

    start_time = time.time()
    results = model(original_path, device="cpu")
    annotated_frame = results[0].plot()
    Image.fromarray(annotated_frame).save(predicted_path)

    # --- Persist session ---
    save_prediction_session(db, uid, original_path, predicted_path, username)

    # --- Persist detections & build label list ---
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
        "time_took": round(time.time() - start_time, 2),
    }
