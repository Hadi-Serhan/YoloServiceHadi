from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from db import get_db
from services.count_service import get_prediction_count_service
from auth import get_current_username


router = APIRouter()


@router.get("/predictions/count")
def get_prediction_count(
    username: str = Depends(get_current_username), db: Session = Depends(get_db)
):
    return get_prediction_count_service(username, db)
