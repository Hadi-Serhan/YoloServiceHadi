# services/predict_service.py
import uuid
import time
import tempfile
import pathlib
from typing import Optional

from PIL import Image
from ultralytics import YOLO
import secrets

from services.validators import (
    validate_mime_and_ext,  # only used when a file is uploaded
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

# NEW: use S3 helpers
from s3_utils import (
    save_original_from_bytes,
    save_predicted_from_file,
    download_to_path,
    build_original_key,
    exists,
    guess_content_type,
)

model = YOLO("yolov8n.pt")  # Load once

MAX_BYTES = 10 * 1024 * 1024  # 10 MB
CHUNK = 1 * 1024 * 1024  # 1 MB


def _read_upload_to_bytes_with_cap(upload_file) -> bytes:
    """Stream a FastAPI UploadFile to memory with a hard cap, then return bytes."""
    total = 0
    chunks = []
    while True:
        chunk = upload_file.file.read(CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_BYTES:
            raise _http_413()
        chunks.append(chunk)
    return b"".join(chunks)


def _http_413():
    # helper to raise HTTP 413 from a non-FastAPI file
    from fastapi import HTTPException

    return HTTPException(status_code=413, detail="File too large (max 10MB)")


def _http_400(msg: str):
    from fastapi import HTTPException

    return HTTPException(status_code=400, detail=msg)


def _http_404(msg: str):
    from fastapi import HTTPException

    return HTTPException(status_code=404, detail=msg)


def process_prediction(
    db,
    chat_id: str,
    file=None,  # Optional[UploadFile]
    img: Optional[str] = None,  # S3 key or bare filename
    username: Optional[str] = None,
    password: Optional[str] = None,
):
    # ----- Auth / quota (unchanged) -----
    if username:
        enforce_db_quota(db, username, monthly_limit=100)
        user = get_user(db, username)
        if user is None:
            create_user(db, username, password)
        elif not secrets.compare_digest(user.password, password):
            raise ValueError("Invalid credentials")

    # ----- Validate input mode -----
    if (file is None) and (img is None):
        raise _http_400(
            "Provide either a multipart 'file' or ?img=<s3_key_or_filename>"
        )
    if (file is not None) and (img is not None):
        raise _http_400("Provide only one of: 'file' or 'img'")

    uid = str(uuid.uuid4())
    start_time = time.time()

    # We'll always run YOLO on a local temp file path.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        input_path = tmpdir / "in_image"
        output_path = tmpdir / "out.png"  # YOLO annotated output (we'll write PNG)

        # ----- Mode A: client uploaded a file -----
        if file is not None:
            # MIME/ext validation based on the uploaded file header & filename
            validate_mime_and_ext(file)

            safe_name = sanitize_filename(file.filename or "upload.jpg")
            # Stream to memory with 10MB cap
            try:
                data = _read_upload_to_bytes_with_cap(file)
            except Exception as e:
                raise e  # already HTTP 413

            # Sniff first ~64KB to ensure it's an actual image
            sniff_image_or_415(data[: 64 * 1024])

            # Upload ORIGINAL to S3 under <chat_id>/original/<basename>
            original_key = save_original_from_bytes(
                chat_id=chat_id,
                filename=safe_name,
                data=data,
                content_type=file.content_type or guess_content_type(safe_name),
            )

            # Also write to local temp file for YOLO
            input_suffix = pathlib.Path(safe_name).suffix or ".jpg"
            input_path = input_path.with_suffix(input_suffix)
            input_path.write_bytes(data)

            # We'll use the original filename as a base for predicted key's name
            preferred_pred_name = pathlib.Path(safe_name).name

        # ----- Mode B: user pointed at an S3 key (or bare filename) -----
        else:
            # normalize to <chat_id>/original/<filename> if user passed just "dog.jpg"
            key = img if "/" in img else build_original_key(chat_id, img)
            # Optional: verify it exists
            if not exists(key):
                raise _http_404(f"S3 object not found: {key}")
            original_key = key

            # Download to temp for YOLO; preserve suffix for PIL decoding
            input_suffix = pathlib.Path(key).suffix or ".jpg"
            input_path = input_path.with_suffix(input_suffix)
            download_to_path(original_key, str(input_path))

            # sanity sniff
            sniff_image_or_415(input_path.read_bytes()[: 64 * 1024])

            preferred_pred_name = pathlib.Path(key).name

        # ----- Run YOLO on local file -----
        results = model(str(input_path), device="cpu")
        annotated_frame = results[0].plot()
        Image.fromarray(annotated_frame).save(output_path)

        # ----- Upload PREDICTED image to S3 -----
        predicted_key = save_predicted_from_file(
            chat_id=chat_id,
            local_path=str(output_path),
            preferred_name=preferred_pred_name,  # so key is <stem>-<uuid>.<ext>
        )

    # ----- Persist session + detections -----
    # Store S3 references (you can store plain keys or 's3://bucket/key'—keys are fine)
    save_prediction_session(
        db,
        uid,
        original_key,
        predicted_key,
        username,
    )

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
        "s3": {
            "original_key": original_key,
            "predicted_key": predicted_key,
        },
    }
