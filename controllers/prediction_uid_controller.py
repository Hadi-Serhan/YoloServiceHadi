from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from db import get_db
from services.prediction_uid_service import get_prediction_by_uid_service
from auth import get_current_username

router = APIRouter()


@router.get("/prediction/{uid}")
def get_prediction(
    uid: str,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_username),
):
    return get_prediction_by_uid_service(uid, username, db)
