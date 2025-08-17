from fastapi import APIRouter, Depends
from db import get_db
from sqlalchemy.orm import Session
from services.delete_service import delete_prediction_service
from auth import get_current_username

router = APIRouter()

@router.delete("/prediction/{uid}")
def delete_prediction(uid: str, username: str = Depends(get_current_username), db: Session = Depends(get_db)):
    return delete_prediction_service(uid, username, db)
