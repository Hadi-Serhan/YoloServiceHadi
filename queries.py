import datetime
from sqlalchemy.orm import Session
from models import PredictionSession, User, DetectionObject


def query_prediction_by_uid(db: Session, uid: str):
    result = db.query(PredictionSession).filter_by(uid=uid).first()
    return result


def get_user(db: Session, username: str):
    return db.query(User).filter_by(username=username).first()


def create_user(db: Session, username: str, password: str):
    user = User(username=username, password=password)
    db.add(user)
    db.commit()


def save_prediction_session(
    db: Session, uid: str, original_path: str, predicted_path: str, username: str
):
    session = PredictionSession(
        uid=uid,
        original_image=original_path,
        predicted_image=predicted_path,
        username=username,
    )
    db.add(session)
    db.commit()


def save_detection_object(
    db: Session, prediction_uid: str, label: str, score: float, box: str
):
    obj = DetectionObject(
        prediction_uid=prediction_uid, label=label, score=score, box=box
    )
    db.add(obj)
    db.commit()


def get_predictions_by_label(db: Session, label: str, username: str):
    rows = (
        db.query(PredictionSession.uid, PredictionSession.timestamp)
        .join(DetectionObject, PredictionSession.uid == DetectionObject.prediction_uid)
        .filter(DetectionObject.label == label, PredictionSession.username == username)
        .distinct()
        .all()
    )
    return rows


def get_predictions_by_score(db: Session, min_score: float, username: str):
    rows = (
        db.query(
            PredictionSession.uid, PredictionSession.timestamp, DetectionObject.score
        )
        .join(DetectionObject, PredictionSession.uid == DetectionObject.prediction_uid)
        .filter(
            DetectionObject.score >= min_score, PredictionSession.username == username
        )
        .all()
    )
    return rows


def user_owns_image(db: Session, image_path: str, column: str, username: str):
    filter_kwargs = {column: image_path, "username": username}
    return db.query(PredictionSession).filter_by(**filter_kwargs).first()


def count_predictions_in_last_week(db: Session, username: str, since: datetime):
    return (
        db.query(PredictionSession)
        .filter(
            PredictionSession.timestamp >= since, PredictionSession.username == username
        )
        .count()
    )


def get_recent_labels(db: Session, username: str, since: datetime):
    subquery = (
        db.query(PredictionSession.uid)
        .filter(
            PredictionSession.timestamp >= since, PredictionSession.username == username
        )
        .subquery()
    )

    rows = (
        db.query(DetectionObject.label)
        .filter(DetectionObject.prediction_uid.in_(subquery))
        .distinct()
        .all()
    )

    return [label for (label,) in rows]


def get_prediction_image_path(db: Session, uid: str, username: str) -> str | None:
    result = (
        db.query(PredictionSession.predicted_image)
        .filter_by(uid=uid, username=username)
        .first()
    )
    return result[0] if result else None


def get_prediction_by_uid_and_user(db: Session, uid: str, username: str):
    return db.query(PredictionSession).filter_by(uid=uid, username=username).first()


def delete_detection_objects_by_uid(db: Session, uid: str):
    db.query(DetectionObject).filter_by(prediction_uid=uid).delete()


def delete_prediction_session(db: Session, uid: str, username: str):
    db.query(PredictionSession).filter_by(uid=uid, username=username).delete()


def count_recent_predictions(db: Session, username: str, since: datetime) -> int:
    return (
        db.query(PredictionSession)
        .filter(
            PredictionSession.timestamp >= since, PredictionSession.username == username
        )
        .count()
    )


def get_detection_objects_for_recent_predictions(
    db: Session, username: str, since: datetime
):
    subquery = (
        db.query(PredictionSession.uid)
        .filter(
            PredictionSession.timestamp >= since, PredictionSession.username == username
        )
        .subquery()
    )

    return (
        db.query(DetectionObject.label, DetectionObject.score)
        .filter(DetectionObject.prediction_uid.in_(subquery))
        .all()
    )
