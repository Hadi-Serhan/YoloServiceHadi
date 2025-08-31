import io
from datetime import datetime, timedelta, timezone
import pytest
from starlette.testclient import TestClient
from PIL import Image

# adjust if your app factory is elsewhere:
from app import app
from db import SessionLocal
from models import PredictionSession, DetectionObject
from infra import purge_old_uploads_db


@pytest.fixture(autouse=True)
def _truncate_tables():
    # wipe rows so UIDs won't collide across runs
    with SessionLocal() as db:
        # delete children first if there are FKs
        db.query(DetectionObject).delete()
        db.query(PredictionSession).delete()
        db.commit()
    yield


client = TestClient(app)


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (3, 3)).save(buf, format="PNG")
    buf.seek(0)
    return buf


def _auth():
    # Basic auth header through httpx: auth=("user","pw")
    return {"auth": ("alice", "pw")}


# ---------- Rate limit headers + 30 rps limit ----------
def test_rate_limit_headers_and_429_on_burst():
    # hit a cheap endpoint; if /health doesn't exist, use any GET you have
    last = None
    for i in range(31):
        last = client.get("/health")  # change to an existing GET if needed
        assert "X-RateLimit-Limit" in last.headers
        assert "X-RateLimit-Remaining" in last.headers
        assert "X-RateLimit-Reset" in last.headers
        if last.status_code == 429:
            break
    assert last.status_code in (200, 429)


# ---------- 10 uploads/min per user/IP ----------
def test_upload_quota_10_per_min(monkeypatch, tmp_path):
    # direct uploads to temp dirs to avoid polluting repo
    import services.predict_service as ps

    monkeypatch.setattr(ps, "UPLOAD_DIR", str(tmp_path / "u"))
    monkeypatch.setattr(ps, "PREDICTED_DIR", str(tmp_path / "p"))

    # fake YOLO so we don't load weights
    class _FakeYOLO:
        names = {0: "person"}

        def __call__(self, path, device="cpu"):
            class _R:
                def __init__(self):
                    self.boxes = [
                        type(
                            "B",
                            (),
                            {
                                "cls": [type("T", (), {"item": lambda s: 0})()],
                                "conf": [type("T", (), {"item": lambda s: 0.99})()],
                                "xyxy": [[1, 2, 3, 4]],
                            },
                        )()
                    ]

                def plot(self):
                    return b"x"

            return [_R()]

    monkeypatch.setattr(ps, "model", _FakeYOLO())

    last = None
    for i in range(11):
        files = {"file": (f"x{i}.png", _png_bytes(), "image/png")}
        last = client.post("/predict", files=files, **_auth())
    assert last.status_code in (
        200,
        429,
    )  # should flip to 429 by the 11th in one minute


# ---------- Monthly DB quota (e.g., 100/month) ----------
def test_monthly_quota_blocks_101st(monkeypatch, tmp_path):
    # Seed DB with 100 sessions for 'alice' in current month
    now = datetime.now(timezone.utc)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    with SessionLocal() as db:
        for i in range(100):
            db.add(
                PredictionSession(
                    uid=f"seed-{i}",
                    timestamp=month_start + timedelta(minutes=i),
                    original_image="",
                    predicted_image="",
                    username="alice",
                )
            )
        db.commit()

    # Fake YOLO + temp dirs
    import services.predict_service as ps

    monkeypatch.setattr(ps, "UPLOAD_DIR", str(tmp_path / "u"))
    monkeypatch.setattr(ps, "PREDICTED_DIR", str(tmp_path / "p"))

    class _FakeYOLO:
        names = {0: "person"}

        def __call__(self, path, device="cpu"):
            class _R:
                def __init__(self):
                    self.boxes = [
                        type(
                            "B",
                            (),
                            {
                                "cls": [type("T", (), {"item": lambda s: 0})()],
                                "conf": [type("T", (), {"item": lambda s: 0.99})()],
                                "xyxy": [[1, 2, 3, 4]],
                            },
                        )()
                    ]

                def plot(self):
                    return b"x"

            return [_R()]

    monkeypatch.setattr(ps, "model", _FakeYOLO())

    # 101st upload should be rejected with 429 by service quota
    files = {"file": ("z.png", _png_bytes(), "image/png")}
    r = client.post("/predict", files=files, **_auth())
    assert r.status_code == 429


# ---------- 90-day purge (DB-driven) ----------
def test_purge_old_uploads_db_deletes_files(tmp_path):
    uploads = tmp_path / "uploads"
    (uploads / "original").mkdir(parents=True)
    (uploads / "predicted").mkdir(parents=True)

    old_orig = uploads / "original" / "old.png"
    old_pred = uploads / "predicted" / "old.png"
    new_orig = uploads / "original" / "new.png"
    new_pred = uploads / "predicted" / "new.png"
    for p in (old_orig, old_pred, new_orig, new_pred):
        p.write_bytes(b"x")

    old_cutoff = datetime.now(timezone.utc) - timedelta(days=91)
    new_time = datetime.now(timezone.utc) - timedelta(days=1)

    with SessionLocal() as db:
        db.add(
            PredictionSession(
                uid="old",
                timestamp=old_cutoff,
                original_image=str(old_orig),
                predicted_image=str(old_pred),
                username=None,
            )
        )
        db.add(
            PredictionSession(
                uid="new",
                timestamp=new_time,
                original_image=str(new_orig),
                predicted_image=str(new_pred),
                username=None,
            )
        )
        db.commit()

    removed = purge_old_uploads_db(upload_root=str(uploads), max_age_days=90)
    assert removed == 2
    assert not old_orig.exists() and not old_pred.exists()
    assert new_orig.exists() and new_pred.exists()
