from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
from fastapi.responses import FileResponse
from db import get_db
from services.image_service import get_image_path_and_validate, get_prediction_image_service
from auth import get_current_username

router = APIRouter()

@router.get("/image/{image_type}/{filename}")
def get_image_route(image_type: str, filename: str, username: str = Depends(get_current_username), db: Session = Depends(get_db)):
    path = get_image_path_and_validate(image_type, filename, username, db)
    return FileResponse(path)


@router.get("/prediction/{uid}/image")
def get_prediction_image(uid: str, request: Request, username: str = Depends(get_current_username), db: Session = Depends(get_db)):
    return get_prediction_image_service(uid, username, request, db)