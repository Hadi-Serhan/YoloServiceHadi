from fastapi import HTTPException
from sqlalchemy.orm import Session
from queries import get_predictions_by_score

def get_predictions_by_score_service(min_score: float, username: str, db: Session):
    if not (0 <= min_score <= 1):
        raise HTTPException(status_code=400, detail="Score must be between 0 and 1")

    rows = get_predictions_by_score(db, min_score, username)
    return [{"uid": uid, "timestamp": timestamp, "score": score} for uid, timestamp, score in rows]
