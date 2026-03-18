import math
import time
import array
from typing import List, Dict, Any
from storage import StorageManager
from indexer import embed_text
from feedback import log_event, RetrievalEvent

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    if not magnitude1 or not magnitude2:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

def retrieve(query: str, db: StorageManager, top_k: int = 10) -> List[Dict[str, Any]]:
    """Ranked retrieval using similarity + recency + success score."""
    # 1. Embed query
    query_vector = embed_text(query)

    # 2. Fetch active chunks + embeddings
    active_chunks = db.fetch_active_chunks()
    if not active_chunks:
        return []

    current_time = time.time()
    decay_lambda = 1e-5

    results = []
    for chunk in active_chunks:
        # Convert BLOB back to vector
        chunk_vector = list(array.array('f', chunk['vector']))

        # 3. Compute similarity
        # Normalized given stub embeddings are [0, 1]
        similarity = sum(a * b for a, b in zip(query_vector, chunk_vector))
        mag_q = math.sqrt(sum(a*a for a in query_vector))
        mag_c = math.sqrt(sum(a*a for a in chunk_vector))
        if mag_q and mag_c:
            similarity /= (mag_q * mag_c)
        else:
            similarity = 0.0

        # 4. Compute recency: exp(-lambda * age_seconds)
        # Handle SQLite TIMESTAMP strings or potential numeric timestamps
        last_time_str = chunk['last_accessed'] or chunk['created_at']
        if isinstance(last_time_str, (int, float)):
             last_time = last_time_str
        else:
            try:
                import datetime
                # Handle possible formats: YYYY-MM-DD HH:MM:SS or YYYY-MM-DDTHH:MM:SS.mmmmmm
                if 'T' in last_time_str:
                    last_time = datetime.datetime.fromisoformat(last_time_str).timestamp()
                else:
                    last_time = datetime.datetime.strptime(last_time_str, '%Y-%m-%d %H:%M:%S').timestamp()
            except (ValueError, TypeError):
                last_time = current_time

        age_seconds = max(0, current_time - last_time)
        recency = math.exp(-decay_lambda * age_seconds)

        # 5. Success score
        success_score = chunk['success_score'] # Default is 1.0 in DB

        # 6. Final Score: (sim * 0.7) + (recency * 0.2) + (success_score * 0.1)
        score = (similarity * 0.7) + (recency * 0.2) + (success_score * 0.1)

        results.append({
            'chunk_id': chunk['id'],
            'content': chunk['content'],
            'score': score,
            'similarity': similarity,
            'recency': recency,
            'success_score': success_score
        })

    # 7. Ranking Pipeline: sort descending
    results.sort(key=lambda x: x['score'], reverse=True)
    top_results = results[:top_k]

    # 8. Feedback loop Integration: every query produces an event
    retrieved_ids = [r['chunk_id'] for r in top_results]
    event = RetrievalEvent(query, retrieved_ids, []) # selected_chunk_ids empty by default
    log_event(event, db)

    return top_results
