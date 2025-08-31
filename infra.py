# infra.py
import os
import time
import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta
from fastapi import Request, HTTPException
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint


# ---------- Rate limiting middleware (burst: 30 rps; uploads: 10/min) ----------
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rps_limit=None, uploads_per_min=None):
        super().__init__(app)
        self.rps_limit = int(os.getenv("RPS_LIMIT", str(rps_limit or 30)))
        self.uploads_per_min = int(
            os.getenv("UPLOADS_PER_MIN", str(uploads_per_min or 10))
        )
        self._req_log = defaultdict(deque)  # key -> deque[timestamps] (1s window)
        self._up_log = defaultdict(deque)  # key -> deque[timestamps] (60s window)

    def _key(self, request: Request) -> str:
        # key by Authorization header if present; otherwise by client IP
        auth = request.headers.get("authorization")
        return auth if auth else f"anon:{request.client.host}"

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        key = self._key(request)
        now = time.time()

        # --- per-user 30 req/sec ---
        q = self._req_log[key]
        while q and now - q[0] > 1.0:
            q.popleft()
        if len(q) >= self.rps_limit:
            reset = max(0.0, 1.0 - (now - q[0]))
            return Response(
                "Rate limit exceeded",
                status_code=429,
                headers={
                    "X-RateLimit-Limit": str(self.rps_limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(reset)),
                },
            )
        q.append(now)

        # --- per-user uploads: 10/min on POST /predict ---
        if (
            request.method.upper() == "POST"
            and request.url.path.rstrip("/") == "/predict"
        ):
            up = self._up_log[key]
            while up and now - up[0] > 60.0:
                up.popleft()
            if len(up) >= self.uploads_per_min:
                reset = max(0.0, 60.0 - (now - up[0]))
                return Response(
                    "Upload quota exceeded (10/min)",
                    status_code=429,
                    headers={
                        "X-RateLimit-Limit": str(self.rps_limit),
                        "X-RateLimit-Remaining": str(max(0, self.rps_limit - len(q))),
                        "X-RateLimit-Reset": str(int(reset)),
                    },
                )
            up.append(now)

        # normal flow
        resp = await call_next(request)
        remaining = max(0, self.rps_limit - len(self._req_log[key]))
        resp.headers["X-RateLimit-Limit"] = str(self.rps_limit)
        resp.headers["X-RateLimit-Remaining"] = str(remaining)
        resp.headers["X-RateLimit-Reset"] = "60"
        return resp


# ---------- 24h cache helper (optional, use in /prediction/{uid}) ----------
class PredictionCache:
    def __init__(self, ttl=24 * 3600):
        self.ttl = ttl
        self._data = {}  # uid -> {"t": ts, "payload": dict}

    def get(self, uid: str):
        now = time.time()
        entry = self._data.get(uid)
        if not entry:
            return None
        if now - entry["t"] > self.ttl:
            self._data.pop(uid, None)
            return None
        return entry["payload"]

    def set(self, uid: str, payload: dict):
        self._data[uid] = {"t": time.time(), "payload": payload}


# ---------- helper: coerce mock counts to int ----------
def _as_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0


# ---------- DB-driven quotas (monthly / 24h) ----------
def enforce_db_quota(
    db,
    username: str,
    *,
    monthly_limit: int | None = None,
    last_24h_limit: int | None = None,
):
    """
    Raise 429 if user exceeded quotas, based on PredictionSession.timestamp.
    Call this inside your /predict service ONLY for authenticated users.
    """
    from models import PredictionSession  # local import to avoid cycles

    # NOTE: your model uses naive UTC datetimes (default=datetime.utcnow)
    now = datetime.utcnow()

    if monthly_limit is not None:
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        count_m = (
            db.query(PredictionSession)
            .filter(
                PredictionSession.username == username,
                PredictionSession.timestamp >= month_start,
            )
            .count()
        )
        if _as_int(count_m) >= int(monthly_limit):
            raise HTTPException(
                status_code=429, detail="Monthly prediction quota exceeded"
            )

    if last_24h_limit is not None:
        day_start = now - timedelta(hours=24)
        count_d = (
            db.query(PredictionSession)
            .filter(
                PredictionSession.username == username,
                PredictionSession.timestamp >= day_start,
            )
            .count()
        )
        if _as_int(count_d) >= int(last_24h_limit):
            raise HTTPException(status_code=429, detail="24h prediction quota exceeded")


# ---------- 90-day uploads cleanup (DB-driven; uses PredictionSession.timestamp) ----------
def purge_old_uploads_db(upload_root: str = "uploads", max_age_days: int = 90) -> int:
    """
    Delete original/predicted image files for sessions older than max_age_days.
    Uses DB timestamp as the source of truth.
    """
    from db import SessionLocal
    from models import (
        PredictionSession,
    )  # uid, timestamp, original_image, predicted_image

    # NOTE: keeping naive UTC to match your model columns
    cutoff = datetime.utcnow() - timedelta(days=max_age_days)
    removed = 0

    def _is_under(root: str, path: str) -> bool:
        root_abs = os.path.realpath(root)
        path_abs = os.path.realpath(path or "")
        return path_abs.startswith(root_abs + os.sep)

    with SessionLocal() as db:
        BATCH = 500
        q = db.query(
            PredictionSession.original_image, PredictionSession.predicted_image
        ).filter(PredictionSession.timestamp < cutoff)
        offset = 0
        while True:
            rows = q.limit(BATCH).offset(offset).all()
            if not rows:
                break
            for orig, pred in rows:
                for p in (orig, pred):
                    try:
                        if p and _is_under(upload_root, p) and os.path.exists(p):
                            os.remove(p)
                            removed += 1
                    except OSError:
                        pass
            offset += BATCH

    return removed


def schedule_daily_cleanup(app, base_dir="uploads", max_age_days=90):
    @app.on_event("startup")
    async def _start():
        async def _loop():
            while True:
                try:
                    # DB-driven is authoritative (files older than the session age)
                    purge_old_uploads_db(
                        upload_root=base_dir, max_age_days=max_age_days
                    )
                finally:
                    await asyncio.sleep(24 * 3600)

        asyncio.create_task(_loop())


# (optional) fallback: purely filesystem mtime sweep
def purge_old_uploads(base_dir: str, max_age_days: int = 90) -> int:
    removed = 0
    cutoff = time.time() - max_age_days * 24 * 3600
    for root, _dirs, files in os.walk(base_dir):
        for name in files:
            path = os.path.join(root, name)
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    removed += 1
            except OSError:
                pass
    return removed
