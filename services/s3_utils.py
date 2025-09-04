# s3_utils.py
from __future__ import annotations

import io
import os
import uuid
import pathlib
import mimetypes
from typing import Optional, List

import boto3
from botocore.exceptions import ClientError

# ---- Configuration from environment ----
AWS_REGION = os.getenv("AWS_REGION")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")

AWS_S3_SSE = os.getenv("AWS_S3_SSE")  # "AES256" or "aws:kms"
AWS_S3_SSE_KMS_KEY_ID = os.getenv("AWS_S3_SSE_KMS_KEY_ID")


# Lazily create a single S3 client. With an EC2 instance role, boto3
# auto-discovers temporary creds via IMDS — no access keys needed.
_s3_client = None


def s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = (
            boto3.client("s3", region_name=AWS_REGION)
            if AWS_REGION
            else boto3.client("s3")
        )
    return _s3_client


def _require_bucket():
    if not AWS_S3_BUCKET:
        raise RuntimeError(
            "AWS_S3_BUCKET is not set. Provide it via environment variable."
        )


# -------------------- Key helpers --------------------


def guess_content_type(filename: str) -> str:
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"


def build_original_key(chat_id: str, filename: str) -> str:
    """<chat_id>/original/<basename>"""
    return (
        f"{chat_id.strip().strip('/')}/original/{pathlib.PurePosixPath(filename).name}"
    )


def build_predicted_key(
    chat_id: str, suggested_name: Optional[str] = None, ext: str = ".png"
) -> str:
    """
    <chat_id>/predicted/<stem>-<uuid>.ext
    If suggested_name is provided, its stem is used; otherwise just a UUID name.
    """
    base = pathlib.PurePosixPath(suggested_name).stem if suggested_name else "pred"
    return f"{chat_id.strip().strip('/')}/predicted/{base}-{uuid.uuid4().hex}{_ensure_dot(ext)}"


def _ensure_dot(extension: str) -> str:
    return extension if extension.startswith(".") else f".{extension}"


# -------------------- Upload helpers --------------------


def _extra_args(
    content_type: Optional[str], metadata: Optional[dict]
) -> Optional[dict]:
    extra = {}
    if content_type:
        extra["ContentType"] = content_type
    if metadata:
        extra["Metadata"] = metadata
    if AWS_S3_SSE:
        extra["ServerSideEncryption"] = AWS_S3_SSE
        if AWS_S3_SSE == "aws:kms" and AWS_S3_SSE_KMS_KEY_ID:
            extra["SSEKMSKeyId"] = AWS_S3_SSE_KMS_KEY_ID
    return extra or None


def upload_bytes(
    data: bytes,
    key: str,
    content_type: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> None:
    """Upload in-memory bytes (great for FastAPI UploadFile.read())."""
    _require_bucket()
    fileobj = io.BytesIO(data)
    s3().upload_fileobj(
        Fileobj=fileobj,
        Bucket=AWS_S3_BUCKET,
        Key=key,
        ExtraArgs=_extra_args(content_type, metadata),
    )


def upload_file(
    path: str,
    key: str,
    content_type: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> None:
    """
    Upload a local filesystem path. Uses multipart automatically for big files.
    Prefer this after your YOLO code writes an annotated image to disk.
    """
    _require_bucket()
    s3().upload_file(
        Filename=path,
        Bucket=AWS_S3_BUCKET,
        Key=key,
        ExtraArgs=_extra_args(content_type, metadata),
    )


# -------------------- Download helpers --------------------


def download_to_path(key: str, dest_path: str) -> None:
    _require_bucket()
    pathlib.Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    s3().download_file(AWS_S3_BUCKET, key, dest_path)


def download_bytes(key: str) -> bytes:
    _require_bucket()
    resp = s3().get_object(Bucket=AWS_S3_BUCKET, Key=key)
    return resp["Body"].read()


# -------------------- Utility ops --------------------


def exists(key: str) -> bool:
    _require_bucket()
    try:
        s3().head_object(Bucket=AWS_S3_BUCKET, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return False if code == 404 else False


def delete_object(key: str) -> None:
    _require_bucket()
    s3().delete_object(Bucket=AWS_S3_BUCKET, Key=key)


def copy_object(src_key: str, dest_key: str) -> None:
    _require_bucket()
    s3().copy(
        CopySource={"Bucket": AWS_S3_BUCKET, "Key": src_key},
        Bucket=AWS_S3_BUCKET,
        Key=dest_key,
        ExtraArgs=_extra_args(None, None),
    )


def list_prefix(prefix: str, max_keys: int = 1000) -> List[str]:
    """
    List object keys under a prefix (non-recursive listing semantics by S3).
    """
    _require_bucket()
    keys: List[str] = []
    paginator = s3().get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=AWS_S3_BUCKET, Prefix=prefix, MaxKeys=max_keys
    ):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def presigned_get_url(key: str, expires_in: int = 3600) -> str:
    _require_bucket()
    return s3().generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": AWS_S3_BUCKET, "Key": key},
        ExpiresIn=expires_in,
    )


# -------------------- High-level convenience --------------------


def save_original_from_bytes(
    chat_id: str, filename: str, data: bytes, content_type: Optional[str] = None
) -> str:
    """
    Stores the uploaded 'original' image under <chat_id>/original/<filename> and returns the S3 key.
    """
    key = build_original_key(chat_id, filename)
    upload_bytes(data, key, content_type or guess_content_type(filename))
    return key


def save_predicted_from_file(
    chat_id: str, local_path: str, preferred_name: Optional[str] = None
) -> str:
    """
    Stores the YOLO-annotated 'predicted' image under <chat_id>/predicted/<stem>-<uuid>.ext and returns the S3 key.
    """
    ext = pathlib.Path(local_path).suffix or ".png"
    key = build_predicted_key(chat_id, suggested_name=preferred_name, ext=ext)
    upload_file(
        local_path,
        key,
        content_type=(
            "image/png" if ext.lower() == ".png" else guess_content_type(local_path)
        ),
    )
    return key
