# services/validators.py
import os
from fastapi import HTTPException, UploadFile
from PIL import Image
import io

ALLOWED_MIMES = {"image/jpeg", "image/png", "image/jpg"}  # include common alias
ALLOWED_EXTS  = {".jpg", ".jpeg", ".png"}

# map extension -> canonical MIME
EXT_TO_MIME = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
}

def sanitize_filename(name: str) -> str:
    name = (name or "").replace("\\", "/")
    return os.path.basename(name)

def validate_mime_and_ext(file: UploadFile):
    ct = (file.content_type or "").lower()
    if ct not in ALLOWED_MIMES:
        raise HTTPException(status_code=415, detail="Only JPEG/PNG supported")

    _, ext = os.path.splitext(file.filename or "")
    ext = ext.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=415, detail="Only .jpg/.jpeg/.png files allowed")

    # Enforce MIME matches extension (treat image/jpg as image/jpeg)
    canonical_ct = "image/jpeg" if ct in {"image/jpeg", "image/jpg"} else ct
    expected_ct = EXT_TO_MIME[ext]
    if canonical_ct != expected_ct:
        raise HTTPException(status_code=415, detail="MIME type does not match file extension")

def sniff_image_or_415(raw_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.verify()
    except Exception:
        raise HTTPException(status_code=415, detail="Invalid or corrupted image")
