from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from queries import (
    count_recent_predictions,
    get_detection_objects_for_recent_predictions,
)
from collections import Counter


def get_stats_service(username: str, db: Session):
    one_week_ago = datetime.now() - timedelta(days=7)

    total_predictions = count_recent_predictions(db, username, one_week_ago)
    detections = get_detection_objects_for_recent_predictions(
        db, username, one_week_ago
    )

    scores = [row.score for row in detections]
    labels = [row.label for row in detections]

    avg_confidence = round(sum(scores) / len(scores), 4) if scores else 0.0
    label_counts = Counter(labels)

    return {
        "total_predictions": total_predictions,
        "average_confidence_score": avg_confidence,
        "most_common_labels": dict(label_counts),
    }
