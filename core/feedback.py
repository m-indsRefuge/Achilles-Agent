import time
from typing import List, Optional, Dict
from storage import StorageManager

# Standardized Signal Reliability Weights
SIGNAL_WEIGHTS = {
    "accept": 1.5,
    "click": 1.0,
    "view": 0.3,
    "dismiss": -0.2
}

class RetrievalEvent:
    def __init__(
        self,
        query: str,
        retrieved_chunk_ids: List[str],
        selected_chunk_ids: List[str],
        dismissed_chunk_ids: Optional[List[str]] = None,
        confidence_weights: Optional[Dict[str, float]] = None,
        signal_type: str = "view",
        source_id: str = "default_user"
    ):
        self.query = query
        self.retrieved_chunk_ids = retrieved_chunk_ids
        self.selected_chunk_ids = selected_chunk_ids
        self.dismissed_chunk_ids = dismissed_chunk_ids or []
        self.confidence_weights = confidence_weights or {}
        self.signal_type = signal_type
        self.source_id = source_id
        self.timestamp = time.time()

def log_event(event: RetrievalEvent, db: StorageManager):
    """Log retrieval tracking and update chunk stats with source-aware reliability weighting."""
    # 1. Persist event in DB for observability
    db.insert_retrieval_event(event.query, event.retrieved_chunk_ids, event.selected_chunk_ids, event.dismissed_chunk_ids, source_id=event.source_id)

    # 2. Fetch Source Reliability Multiplier
    source_multiplier = db.get_source_reliability(event.source_id)

    # 3. Update retrieval_stats for each chunk (Calibrated Signal Reinforcement)
    for chunk_id in event.retrieved_chunk_ids:
        # Use explicit weight if provided, otherwise derive from signal type
        if chunk_id in event.confidence_weights:
            weight = event.confidence_weights[chunk_id]
        else:
            weight = SIGNAL_WEIGHTS.get(event.signal_type, 1.0)

        # Apply combined calibration and source reliability
        effective_weight = weight * source_multiplier

        if chunk_id in event.selected_chunk_ids:
            db.update_retrieval_stats(chunk_id, signal="selected", weight=effective_weight)
        elif chunk_id in event.dismissed_chunk_ids:
            db.update_retrieval_stats(chunk_id, signal="dismissed", weight=effective_weight)
        else:
            # Neutral: no score change (ignored)
            db.update_retrieval_stats(chunk_id, signal="neutral")

    # 4. Update Source Reliability based on interaction quality
    is_good = len(event.selected_chunk_ids) > 0
    db.update_source_reliability(event.source_id, is_good)

    # 5. Local Score Normalization: ensures scores remain bounded [0, 1] relative to set
    db.normalize_retrieval_set_scores(event.retrieved_chunk_ids)
