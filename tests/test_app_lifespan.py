from starlette.testclient import TestClient
from app import app


def test_lifespan_runs_startup_and_shutdown(monkeypatch):
    # prove we hit the purge call at startup
    calls = []
    # Point to the symbol as imported in app.py (module path matters!)
    monkeypatch.setattr("app.purge_old_uploads_db", lambda **kw: calls.append(kw))

    # Enter + exit lifespan explicitly
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200

    # If lifespan ran, purge_old_uploads_db was called at least once
    assert calls, "purge_old_uploads_db was not called during startup"
