from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from db import get_db
from services.label_service import (
    get_predictions_by_label_service,
    get_recent_labels_service,
)
from auth import get_current_username

router = APIRouter()


@router.get("/predictions/label/{label}")
def get_predictions_by_label_route(
    label: str,
    username: str = Depends(get_current_username),
    db: Session = Depends(get_db),
):
    return get_predictions_by_label_service(label, username, db)


@router.get("/labels")
def get_labels(
    username: str = Depends(get_current_username), db: Session = Depends(get_db)
):
    return get_recent_labels_service(username, db)
