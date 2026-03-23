import time
from typing import List, Optional, Dict
from storage import StorageManager

class RetrievalEvent:
    def __init__(
        self,
        query: str,
        retrieved_chunk_ids: List[str],
        selected_chunk_ids: List[str],
        dismissed_chunk_ids: Optional[List[str]] = None,
        confidence_weights: Optional[Dict[str, float]] = None
    ):
        self.query = query
        self.retrieved_chunk_ids = retrieved_chunk_ids
        self.selected_chunk_ids = selected_chunk_ids
        self.dismissed_chunk_ids = dismissed_chunk_ids or []
        self.confidence_weights = confidence_weights or {}
        self.timestamp = time.time()

def log_event(event: RetrievalEvent, db: StorageManager):
    """Log retrieval tracking and update chunk stats with selective signals and confidence weighting."""
    # 1. Persist event in DB for observability
    db.insert_retrieval_event(event.query, event.retrieved_chunk_ids, event.selected_chunk_ids, event.dismissed_chunk_ids)

    # 2. Update retrieval_stats for each chunk (Signal-Aware Reinforcement)
    for chunk_id in event.retrieved_chunk_ids:
        weight = event.confidence_weights.get(chunk_id, 1.0)

        if chunk_id in event.selected_chunk_ids:
            db.update_retrieval_stats(chunk_id, signal="selected", weight=weight)
        elif chunk_id in event.dismissed_chunk_ids:
            db.update_retrieval_stats(chunk_id, signal="dismissed", weight=weight)
        else:
            # Neutral: increment count but no score change
            db.update_retrieval_stats(chunk_id, signal="neutral")

    # 3. Local Score Normalization: ensures scores remain bounded [0, 1] relative to set
    db.normalize_retrieval_set_scores(event.retrieved_chunk_ids)
