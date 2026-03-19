import math
import json
from typing import List, Dict
from datetime import datetime

# Context-aware retrieval upgrade module

def cosine_similarity(vec1, vec2):
    if not vec1 or not vec2:
        return 0.0
    min_len = min(len(vec1), len(vec2))
    v1 = vec1[:min_len]
    v2 = vec2[:min_len]
    dot = sum(a*b for a,b in zip(v1,v2))
    norm1 = math.sqrt(sum(a*a for a in v1))
    norm2 = math.sqrt(sum(b*b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def expand_context(chunk, db, window_size=1):
    cursor = db.conn.cursor()
    cursor.execute("SELECT * FROM chunks WHERE document_id=? AND is_active=1 ORDER BY start_line", (chunk["document_id"],))
    rows = cursor.fetchall()

    idx = None
    for i, r in enumerate(rows):
        if r["id"] == chunk["id"]:
            idx = i
            break

    if idx is None:
        return [chunk]

    start = max(0, idx - window_size)
    end = min(len(rows), idx + window_size + 1)
    return rows[start:end]


def stitch_chunks(chunks):
    ordered = sorted(chunks, key=lambda c: c["start_line"])
    return "\n".join(c["content"] for c in ordered)


def compute_recency(ts):
    if not ts:
        return 1.0
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    age = (datetime.utcnow() - ts).total_seconds()
    return math.exp(-1e-5 * age)


def retrieve_with_context(query, db, embed_fn, top_k=5):
    query_vec = embed_fn(query)

    chunks = db.fetch_active_chunks()
    embeddings = {e["chunk_id"]: json.loads(e["vector"]) for e in db.get_embeddings()}

    scored = []

    for c in chunks:
        cid = c["id"]
        if cid not in embeddings:
            continue

        sim = cosine_similarity(query_vec, embeddings[cid])
        stats = None
        try:
            stats = db.get_retrieval_stats(cid)
        except:
            pass

        if stats:
            rec = compute_recency(stats["last_accessed"] or c["created_at"])
            suc = stats["success_score"]
        else:
            rec = 1.0
            suc = 1.0

        score = sim*0.7 + rec*0.2 + suc*0.1
        scored.append((c, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    for chunk, score in scored[:top_k]:
        ctx_chunks = expand_context(chunk, db)
        context = stitch_chunks(ctx_chunks)
        results.append({
            "chunk_id": chunk["id"],
            "score": score,
            "context": context
        })

    return results
