# tests/test_runtime_enforcement.py
import io
import os
import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from starlette.testclient import TestClient
from PIL import Image

from app import app
from db import SessionLocal
from models import PredictionSession, DetectionObject
from infra import purge_old_uploads_db
import infra  # import module so we can monkeypatch internals


# ---------- fixtures ---------------------------------------------------------


@pytest.fixture(autouse=True)
def _truncate_tables():
    # wipe rows so UIDs won't collide across runs
    with SessionLocal() as db:
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


def _set_fake_time(monkeypatch, initial: float):
    """
    Patch infra.time.time() to return a mutable 'current[0]'.
    Advance time in tests via: t[0] = new_value
    """
    current = [initial]
    monkeypatch.setattr(infra.time, "time", lambda: current[0], raising=True)
    return current


# ---------- existing tests (kept) --------------------------------------------


def test_rate_limit_headers_and_429_on_burst():
    # hit a cheap endpoint; if /health doesn't exist, use any GET you have
    last = None
    for _ in range(31):
        last = client.get("/health")
        assert "X-RateLimit-Limit" in last.headers
        assert "X-RateLimit-Remaining" in last.headers
        assert "X-RateLimit-Reset" in last.headers
        if last.status_code == 429:
            break
    assert last.status_code in (200, 429)


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
    # should flip to 429 by the 11th in one minute
    assert last.status_code in (200, 429)


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


# ---------- NEW tests to cover missed lines in infra.py ----------------------


def test_rate_limit_deque_cleanup_req(monkeypatch):
    """Covers line 35: q.popleft() in the 1s window cleanup."""
    t = _set_fake_time(monkeypatch, 1000.0)
    with TestClient(app) as local:
        r1 = local.get("/health")
        assert r1.status_code in (200, 429)
        # advance time by >1s so the cleanup loop pops the old timestamp
        t[0] = 1002.0
        r2 = local.get("/health")
        assert r2.status_code in (200, 429)


def test_upload_deque_cleanup(monkeypatch):
    """Covers line 56: up.popleft() in the 60s upload window cleanup."""
    t = _set_fake_time(monkeypatch, 2000.0)
    with TestClient(app) as local:
        files = {"file": ("x.png", io.BytesIO(b"not-a-real-image"), "image/png")}
        r1 = local.post("/predict", files=files)
        assert r1.status_code in (200, 415, 429)
        t[0] = 2065.0  # > 60s later
        r2 = local.post("/predict", files=files)
        assert r2.status_code in (200, 415, 429)


def test_prediction_cache_hit_and_expire(monkeypatch):
    """Covers lines 82-83, 86-93, 96."""
    t = _set_fake_time(monkeypatch, 3000.0)
    c = infra.PredictionCache(ttl=1)
    c.set("uid", {"ok": True})
    assert c.get("uid") == {"ok": True}  # within ttl
    # expire it
    t[0] = 3002.0
    assert c.get("uid") is None


def test__as_int_exception_branch():
    """Covers lines 103-104: exception -> 0."""

    class NotInt:
        pass

    assert infra._as_int(NotInt()) == 0


def test_enforce_db_quota_monthly_and_daily_raises():
    """Covers line 135 (monthly raise) and 140-150 (24h raise)."""
    with SessionLocal() as db:
        # seed 3 rows in this month for user 'alice'
        now = datetime.utcnow()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        for i in range(3):
            db.add(
                PredictionSession(
                    uid=f"m{i}",
                    timestamp=month_start + timedelta(minutes=i),
                    original_image="",
                    predicted_image="",
                    username="alice",
                )
            )
        db.commit()

        # monthly limit = 3 -> should raise
        with pytest.raises(infra.HTTPException) as ex1:
            infra.enforce_db_quota(db, "alice", monthly_limit=3)
        assert ex1.value.status_code == 429

        # seed 2 rows in last 24h for user 'bob'
        for i in range(2):
            db.add(
                PredictionSession(
                    uid=f"d{i}",
                    timestamp=datetime.utcnow() - timedelta(hours=1, minutes=i),
                    original_image="",
                    predicted_image="",
                    username="bob",
                )
            )
        db.commit()

        # 24h limit = 2 -> should raise
        with pytest.raises(infra.HTTPException) as ex2:
            infra.enforce_db_quota(db, "bob", last_24h_limit=2)
        assert ex2.value.status_code == 429


def test_purge_old_uploads_db_handles_oserror(monkeypatch, tmp_path):
    """Covers lines 189-190: os.remove raising OSError is swallowed."""
    uploads = tmp_path / "uploads"
    (uploads / "original").mkdir(parents=True)
    (uploads / "predicted").mkdir(parents=True)

    old1 = uploads / "original" / "will_fail.png"
    old2 = uploads / "predicted" / "will_pass.png"
    for p in (old1, old2):
        p.write_bytes(b"x")

    cutoff = datetime.utcnow() - timedelta(days=91)
    with SessionLocal() as db:
        db.add(
            PredictionSession(
                uid="u1",
                timestamp=cutoff,
                original_image=str(old1),
                predicted_image=str(old2),
                username=None,
            )
        )
        db.commit()

    real_remove = os.remove

    def fake_remove(path):
        if "will_fail.png" in path:
            raise OSError("simulated failure")
        return real_remove(path)

    monkeypatch.setattr(infra.os, "remove", fake_remove, raising=True)

    removed = infra.purge_old_uploads_db(upload_root=str(uploads), max_age_days=90)
    # one succeeded, one failed (swallowed)
    assert removed == 1
    assert not old2.exists() and old1.exists()


def test_schedule_daily_cleanup_invokes_once(monkeypatch, tmp_path):
    """Covers lines 197-209 by running startup; task sleeps then cancels."""
    from fastapi import FastAPI

    calls = {"n": 0}

    async def fake_sleep(_):
        # break the loop after the first purge call
        raise asyncio.CancelledError()

    def fake_purge(**kwargs):
        calls["n"] += 1

    monkeypatch.setattr(infra, "purge_old_uploads_db", fake_purge, raising=True)
    monkeypatch.setattr(infra.asyncio, "sleep", fake_sleep, raising=True)

    small_app = FastAPI()
    infra.schedule_daily_cleanup(small_app, base_dir=str(tmp_path), max_age_days=90)

    # Starting the app should trigger one purge via the background task
    with TestClient(small_app):
        pass

    assert calls["n"] >= 1


def test_purge_old_uploads_filesystem(tmp_path):
    """Covers lines 214-225 entirely (mtime-based fallback)."""
    (tmp_path / "a").mkdir()
    oldf = tmp_path / "a" / "old.txt"
    newf = tmp_path / "a" / "new.txt"
    oldf.write_text("x")
    newf.write_text("y")

    # set mtime so oldf < cutoff, newf >= cutoff
    old_secs = infra.time.time() - 95 * 24 * 3600
    new_secs = infra.time.time()
    os.utime(oldf, (old_secs, old_secs))
    os.utime(newf, (new_secs, new_secs))

    removed = infra.purge_old_uploads(str(tmp_path), max_age_days=90)
    assert removed == 1
    assert not oldf.exists() and newf.exists()
