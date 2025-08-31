from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import Optional
from sqlalchemy.orm import Session
from services.predict_service import process_prediction
from db import get_db

router = APIRouter()


@router.post("/predict")
def predict(
    file: UploadFile = File(...),
    credentials: Optional[HTTPBasicCredentials] = Depends(HTTPBasic(auto_error=False)),
    db: Session = Depends(get_db),
):
    username = credentials.username if credentials else None
    password = credentials.password if credentials else None

    try:
        return process_prediction(file, db, username, password)
    except ValueError as ve:
        raise HTTPException(status_code=401, detail=str(ve))
