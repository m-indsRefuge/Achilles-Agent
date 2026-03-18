from datetime import datetime
from typing import List

class RetrievalEvent:
    def __init__(self, query: str, retrieved_chunk_ids: List[str], selected_chunk_ids: List[str]):
        self.query = query
        self.retrieved_chunk_ids = retrieved_chunk_ids
        self.selected_chunk_ids = selected_chunk_ids
        self.timestamp = datetime.utcnow()


def log_event(event: RetrievalEvent, db):
    """
    Process retrieval event and update stats.
    """

    for chunk_id in event.retrieved_chunk_ids:
        used = chunk_id in event.selected_chunk_ids
        db.update_retrieval_stats(chunk_id, used)


def apply_feedback_loop(query: str, retrieved_chunks, selected_chunks, db):
    """
    High-level helper to create and log retrieval event.
    """

    retrieved_ids = [chunk["id"] for chunk, _ in retrieved_chunks]
    selected_ids = [chunk["id"] for chunk in selected_chunks]

    event = RetrievalEvent(
        query=query,
        retrieved_chunk_ids=retrieved_ids,
        selected_chunk_ids=selected_ids
    )

    log_event(event, db)

    return event
