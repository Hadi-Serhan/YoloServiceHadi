import os
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from queries import get_prediction_image_path, user_owns_image

def get_image_path_and_validate(image_type: str, filename: str, username: str, db: Session) -> str:
    if image_type not in ["original", "predicted"]:
        raise HTTPException(status_code=400, detail="Invalid image type")

    path = os.path.join("uploads", image_type, filename)

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")

    column = f"{image_type}_image"
    record = user_owns_image(db, path, column, username)

    if not record:
        raise HTTPException(status_code=404, detail="Access denied")

    return path


def get_prediction_image_service(uid: str, username: str, request: Request, db: Session):
    accept = request.headers.get("accept", "")

    image_path = get_prediction_image_path(db, uid, username)
    if not image_path:
        raise HTTPException(status_code=404, detail="Prediction not found")

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Predicted image file not found")

    if "image/png" in accept:
        return FileResponse(image_path, media_type="image/png")
    elif "image/jpeg" in accept or "image/jpg" in accept:
        return FileResponse(image_path, media_type="image/jpeg")
    else:
        raise HTTPException(status_code=406, detail="Client does not accept an image format")