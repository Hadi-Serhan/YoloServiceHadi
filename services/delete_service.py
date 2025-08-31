import os
from fastapi import HTTPException
from sqlalchemy.orm import Session
from queries import (
    get_prediction_by_uid_and_user,
    delete_detection_objects_by_uid,
    delete_prediction_session,
)


def delete_prediction_service(uid: str, username: str, db: Session):
    prediction = get_prediction_by_uid_and_user(db, uid, username)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    # Delete detection objects and session
    delete_detection_objects_by_uid(db, uid)
    delete_prediction_session(db, uid, username)

    # Commit DB changes
    db.commit()

    # Delete associated image files
    for path_key in ["original_image", "predicted_image"]:
        file_path = getattr(prediction, path_key)
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Warning: Failed to delete {file_path} — {e}")

    return {"detail": f"Prediction {uid} deleted successfully."}
