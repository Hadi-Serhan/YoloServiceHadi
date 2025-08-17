from datetime import datetime, timedelta
from fastapi import HTTPException
from queries import get_predictions_by_label, get_recent_labels
from sqlalchemy.orm import Session
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def get_predictions_by_label_service(label: str, username: str, db: Session):
    if label not in model.names.values():
        raise HTTPException(status_code=404, detail="Label not supported")

    rows = get_predictions_by_label(db, label, username)
    return [{"uid": uid, "timestamp": timestamp} for uid, timestamp in rows]


def get_recent_labels_service(username: str, db: Session):
    one_week_ago = datetime.now() - timedelta(days=7)
    labels = get_recent_labels(db, username, one_week_ago)
    return {"labels": labels}