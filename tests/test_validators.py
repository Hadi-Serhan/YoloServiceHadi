# tests/test_validators.py
import io
import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile as StarletteUploadFile, Headers
from PIL import Image

from services.validators import (
    sanitize_filename, validate_mime_and_ext, sniff_image_or_415
)


def test_sanitize_filename_strips_paths():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("C:\\temp\\evil.png") == "evil.png"
    assert sanitize_filename("") == ""

def _upload_file(name: str, mime: str, data: bytes):
    # Starlette UploadFile reads content_type from headers
    return StarletteUploadFile(
        filename=name,
        file=io.BytesIO(data),
        headers=Headers({"content-type": mime}),
    )

def test_validate_mime_and_ext_accepts_jpeg_png():
    for name, ct in [
        ("a.jpg",  "image/jpeg"),
        ("a.jpeg", "image/jpeg"),
        ("a.png",  "image/png"),
    ]:
        uf = _upload_file(name, ct, b"dummy")
        validate_mime_and_ext(uf)  # should not raise

@pytest.mark.parametrize("name,ct", [
    ("a.gif","image/gif"),     # unsupported mime/ext
    ("a.txt","text/plain"),    # unsupported mime/ext
    ("a.jpg","image/png"),     # mime/ext mismatch
    ("a.png","image/jpeg"),    # mime/ext mismatch
    ("a.gif","image/jpeg"),    # unsupported ext
])
def test_validate_mime_and_ext_rejects_invalid(name, ct):
    uf = _upload_file(name, ct, b"dummy")
    with pytest.raises(HTTPException) as ex:
        validate_mime_and_ext(uf)
    assert ex.value.status_code == 415

def test_sniff_image_or_415_valid_png():
    buf = io.BytesIO()
    Image.new("RGB", (2,2), (255,0,0)).save(buf, format="PNG")
    sniff_image_or_415(buf.getvalue())  # no exception

def test_sniff_image_or_415_invalid_bytes():
    with pytest.raises(HTTPException) as ex:
        sniff_image_or_415(b"not an image at all")
    assert ex.value.status_code == 415
