from fastapi import FastAPI
from dotenv import load_dotenv
from db import Base, engine
load_dotenv()
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

app = FastAPI()
@app.on_event("startup")
def create_tables():
    Base.metadata.create_all(bind=engine)

@app.get("/health")
def health():
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
