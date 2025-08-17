from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from queries import count_predictions_in_last_week

def get_prediction_count_service(username: str, db: Session):
    one_week_ago = datetime.now() - timedelta(days=7)
    count = count_predictions_in_last_week(db, username, one_week_ago)
    return {"count": count}
