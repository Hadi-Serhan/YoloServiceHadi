from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import Optional
from sqlalchemy.orm import Session
from services.predict_service import process_prediction
from db import get_db

router = APIRouter()


@router.post("/predict")
def predict(
    chat_id: str = Query(..., description="Logical folder for user/session in S3"),
    img: Optional[str] = Query(
        None,
        description="S3 key or bare filename (e.g., 'dog.jpg' or '123/original/dog.jpg')",
    ),
    file: Optional[UploadFile] = File(None),
    credentials: Optional[HTTPBasicCredentials] = Depends(HTTPBasic(auto_error=False)),
    db: Session = Depends(get_db),
):
    username = credentials.username if credentials else None
    password = credentials.password if credentials else None

    try:
        return process_prediction(
            db=db,
            chat_id=chat_id,
            file=file,
            img=img,
            username=username,
            password=password,
        )
    except ValueError as ve:
        raise HTTPException(status_code=401, detail=str(ve))
