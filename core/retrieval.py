import math
from typing import List, Tuple
from datetime import datetime


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if not vec1 or not vec2:
        return 0.0

    min_len = min(len(vec1), len(vec2))
    v1 = vec1[:min_len]
    v2 = vec2[:min_len]

    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


def embed_query(text: str) -> List[float]:
    # Same deterministic embedding as indexer
    return [float(ord(c)) for c in text[:128]]


def compute_recency(timestamp: datetime, decay_lambda: float = 1e-5) -> float:
    if not timestamp:
        return 1.0

    age_seconds = (datetime.utcnow() - timestamp).total_seconds()
    return math.exp(-decay_lambda * age_seconds)


def retrieve(query: str, db, top_k: int = 10) -> List[Tuple]:
    query_embedding = embed_query(query)

    chunks = db.fetch_active_chunks()
    embeddings = {row["chunk_id"]: row["vector"] for row in db.get_embeddings()}

    scored_results = []

    for chunk in chunks:
        chunk_id = chunk["id"]

        if chunk_id not in embeddings:
            continue

        # Convert stored JSON vector back to list
        vector = eval(embeddings[chunk_id])

        similarity = cosine_similarity(query_embedding, vector)

        stats = db.get_retrieval_stats(chunk_id)

        if stats:
            recency = compute_recency(stats["last_accessed"] or chunk["created_at"])
            success_score = stats["success_score"]
        else:
            recency = 1.0
            success_score = 1.0

        score = (similarity * 0.7) + (recency * 0.2) + (success_score * 0.1)

        scored_results.append((chunk, score))

    scored_results.sort(key=lambda x: x[1], reverse=True)

    return scored_results[:top_k]
