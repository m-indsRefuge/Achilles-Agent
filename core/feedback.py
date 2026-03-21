import time
from typing import List, Optional
from storage import StorageManager

class RetrievalEvent:
    def __init__(self, query: str, retrieved_chunk_ids: List[str], selected_chunk_ids: List[str], dismissed_chunk_ids: Optional[List[str]] = None):
        self.query = query
        self.retrieved_chunk_ids = retrieved_chunk_ids
        self.selected_chunk_ids = selected_chunk_ids
        self.dismissed_chunk_ids = dismissed_chunk_ids or []
        self.timestamp = time.time()

def log_event(event: RetrievalEvent, db: StorageManager):
    """Log retrieval tracking and update chunk stats with selective signals."""
    # 1. Persist event in DB for observability
    db.insert_retrieval_event(event.query, event.retrieved_chunk_ids, event.selected_chunk_ids, event.dismissed_chunk_ids)

    # 2. Update retrieval_stats for each chunk (Signal-Aware Reinforcement)
    for chunk_id in event.retrieved_chunk_ids:
        if chunk_id in event.selected_chunk_ids:
            db.update_retrieval_stats(chunk_id, signal="selected")
        elif chunk_id in event.dismissed_chunk_ids:
            db.update_retrieval_stats(chunk_id, signal="dismissed")
        else:
            # Neutral: increment count but no score change
            db.update_retrieval_stats(chunk_id, signal="neutral")
