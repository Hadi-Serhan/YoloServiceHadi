import asyncio
from fastapi import FastAPI
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from db import Base, engine
from infra import RateLimitMiddleware, purge_old_uploads_db


from controllers import (
    predict_controller,
    prediction_uid_controller,
    label_controller,
    score_controller,
    image_controller,
    count_controller,
    delete_controller,
    stats_controller,
)

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # create tables once at startup
    from models import PredictionSession, DetectionObject, User  # noqa: F401

    Base.metadata.create_all(bind=engine)

    # kick off daily cleanup loop (90-day purge; DB-driven)
    async def _cleanup_loop():
        while True:
            try:
                purge_old_uploads_db(upload_root="uploads", max_age_days=90)
            finally:
                await asyncio.sleep(24 * 3600)

    cleanup_task = asyncio.create_task(_cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()


app = FastAPI(lifespan=lifespan)

# global rate limits + headers
app.add_middleware(RateLimitMiddleware)  # reads RPS_LIMIT / UPLOADS_PER_MIN


@app.get("/health")
def health():  # pragma: no cover
    return {"status": "ok"}


# Register routers
app.include_router(predict_controller.router)
app.include_router(prediction_uid_controller.router)
app.include_router(label_controller.router)
app.include_router(score_controller.router)
app.include_router(image_controller.router)
app.include_router(count_controller.router)
app.include_router(delete_controller.router)
app.include_router(stats_controller.router)

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
