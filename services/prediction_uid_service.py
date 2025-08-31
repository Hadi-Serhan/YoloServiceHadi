from fastapi import HTTPException
from sqlalchemy.orm import Session
from queries import query_prediction_by_uid


def get_prediction_by_uid_service(uid: str, username: str, db: Session):
    prediction = query_prediction_by_uid(db, uid)

    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    if prediction.username != username:
        raise HTTPException(status_code=403, detail="Access denied")

    return {
        "uid": prediction.uid,
        "timestamp": prediction.timestamp,
        "original_image": prediction.original_image,
        "predicted_image": prediction.predicted_image,
    }
