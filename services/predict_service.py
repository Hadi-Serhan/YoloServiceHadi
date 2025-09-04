# services/predict_service.py
import os
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

# ========= Back-compat constants so tests can monkeypatch =========
UPLOAD_DIR = "uploads/original"
PREDICTED_DIR = "uploads/predicted"

# ========= Toggle S3 by env (CI/dev will run local mode) =========
USE_S3 = bool(os.getenv("AWS_S3_BUCKET"))

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
    from fastapi import HTTPException

    return HTTPException(status_code=413, detail="File too large (max 10MB)")


def _http_400(msg: str):
    from fastapi import HTTPException

    return HTTPException(status_code=400, detail=msg)


def _http_404(msg: str):
    from fastapi import HTTPException

    return HTTPException(status_code=404, detail=msg)


# ---------------- S3 helpers are lazy-imported (keeps tests & coverage happy) ----------------
def _s3_prepare_from_upload(
    chat_id: str, safe_name: str, data: bytes
):  # pragma: no cover
    from services.s3_utils import save_original_from_bytes

    original_key = save_original_from_bytes(
        chat_id=chat_id,
        filename=safe_name,
        data=data,
        content_type=None,  # s3_utils will guess if None
    )
    return original_key


def _s3_prepare_from_key(
    chat_id: str, img: str, local_dst: pathlib.Path
):  # pragma: no cover
    from services.s3_utils import build_original_key, download_to_path, exists

    key = img if "/" in img else build_original_key(chat_id, img)
    if not exists(key):
        raise _http_404(f"S3 object not found: {key}")
    download_to_path(key, str(local_dst))
    return key


def _s3_upload_predicted(
    chat_id: str, output_path: pathlib.Path, preferred_name: str
):  # pragma: no cover
    from services.s3_utils import save_predicted_from_file

    return save_predicted_from_file(
        chat_id=chat_id, local_path=str(output_path), preferred_name=preferred_name
    )


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

    # Ensure local dirs exist (used in local mode; harmless otherwise)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(PREDICTED_DIR, exist_ok=True)

    # We'll always run YOLO on a local temp file path.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        input_path = tmpdir / "in_image"
        output_path = tmpdir / "out.png"  # YOLO annotated output (we'll write PNG)

        # ----- Mode A: client uploaded a file -----
        if file is not None:
            validate_mime_and_ext(file)
            safe_name = sanitize_filename(file.filename or "upload.jpg")

            # Stream to memory with 10MB cap and sniff
            data = _read_upload_to_bytes_with_cap(file)
            sniff_image_or_415(data[: 64 * 1024])

            # LOCAL MODE (default in CI): write to UPLOAD_DIR and infer input suffix
            if not USE_S3:
                _, ext = os.path.splitext(safe_name)
                original_path = os.path.join(UPLOAD_DIR, uid + (ext.lower() or ".jpg"))
                with open(original_path, "wb") as out:
                    out.write(data)
                input_path = input_path.with_suffix(ext or ".jpg")
                input_path.write_bytes(data)
                original_ref = original_path  # what we save to DB
                preferred_pred_name = pathlib.Path(safe_name).name
            else:
                # S3 MODE
                original_key = _s3_prepare_from_upload(
                    chat_id, safe_name, data
                )  # pragma: no cover
                input_path = input_path.with_suffix(
                    pathlib.Path(safe_name).suffix or ".jpg"
                )
                input_path.write_bytes(data)
                original_ref = original_key  # what we save to DB
                preferred_pred_name = pathlib.Path(safe_name).name

        # ----- Mode B: user pointed at an S3 key (or bare filename) -----
        else:
            if not USE_S3:
                # In local mode we don't support download-by-key; keep behavior simple for tests
                raise _http_400("img key download requires S3 to be enabled")
            key = _s3_prepare_from_key(
                chat_id, img, input_path.with_suffix(pathlib.Path(img).suffix or ".jpg")
            )  # pragma: no cover
            # sanity sniff
            sniff_image_or_415(pathlib.Path(str(input_path)).read_bytes()[: 64 * 1024])
            original_ref = key
            preferred_pred_name = pathlib.Path(key).name

        # ----- Run YOLO on local file -----
        results = model(str(input_path), device="cpu")
        annotated_frame = results[0].plot()
        Image.fromarray(annotated_frame).save(output_path)

        # ----- Store predicted -----
        if not USE_S3:
            predicted_path = os.path.join(PREDICTED_DIR, uid + ".png")
            Image.fromarray(annotated_frame).save(predicted_path)
            predicted_ref = predicted_path
            s3_block = None
        else:
            predicted_key = _s3_upload_predicted(
                chat_id, output_path, preferred_pred_name
            )  # pragma: no cover
            predicted_ref = predicted_key
            s3_block = {"original_key": original_ref, "predicted_key": predicted_key}

    # ----- Persist session + detections -----
    save_prediction_session(
        db,
        uid,
        original_ref,
        predicted_ref,
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

    resp = {
        "prediction_uid": uid,
        "detection_count": len(results[0].boxes),
        "labels": labels,
        "time_took": round(time.time() - start_time, 2),
    }
    if s3_block:
        resp["s3"] = s3_block
    return resp
