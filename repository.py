from sqlalchemy.orm import Session
from models import PredictionSession

def query_prediction_by_uid(db: Session, uid: str):
    result = db.query(PredictionSession).filter_by(uid=uid).first()
    return result