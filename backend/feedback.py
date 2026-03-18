import time
from typing import List, Optional
from storage import StorageManager

class RetrievalEvent:
    def __init__(self, query: str, retrieved_chunk_ids: List[str], selected_chunk_ids: List[str]):
        self.query = query
        self.retrieved_chunk_ids = retrieved_chunk_ids
        self.selected_chunk_ids = selected_chunk_ids
        self.timestamp = time.time()

def log_event(event: RetrievalEvent, db: StorageManager):
    """Log retrieval tracking and update chunk stats."""
    # success_score += 1.0 if selected, else -0.1 (floor 0.1)
    for chunk_id in event.retrieved_chunk_ids:
        used = (chunk_id in event.selected_chunk_ids)
        db.update_retrieval_stats(chunk_id, used)
