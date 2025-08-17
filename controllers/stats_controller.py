from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from db import get_db
from auth import get_current_username
from services.stats_service import get_stats_service

router = APIRouter()

@router.get("/stats")
def get_stats(username: str = Depends(get_current_username), db: Session = Depends(get_db)):
    return get_stats_service(username, db)
