from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from db import get_db
from auth import get_current_username
from services.score_service import get_predictions_by_score_service

router = APIRouter()


@router.get("/predictions/score/{min_score}")
def get_predictions_by_score_route(
    min_score: float,
    username: str = Depends(get_current_username),
    db: Session = Depends(get_db),
):
    return get_predictions_by_score_service(min_score, username, db)
