import math
import time
import array
import re
from typing import List, Dict, Any, Optional
from storage import StorageManager
from embedding import embed_query
from feedback import log_event, RetrievalEvent
from scoring import RetrievalScorer

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    if not magnitude1 or not magnitude2:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

def normalize_query(query: str) -> str:
    """Ensures semantically equivalent inputs produce identical system behavior."""
    if not query:
        return ""
    query = query.strip().lower()
    return " ".join(query.split())

def compute_dynamic_weights(query: str, db: StorageManager) -> Dict[str, float]:
    """Balances similarity vs feedback based on query context (repetition)."""
    repeat_count = db.get_query_frequency(query)

    # feedback_weight ∈ [0.1, 0.5]
    feedback_weight = min(0.5, 0.1 + 0.1 * repeat_count)
    similarity_weight = 1.0 - feedback_weight

    return {
        "similarity_weight": round(similarity_weight, 2),
        "feedback_weight": round(feedback_weight, 2)
    }

def retrieve(query: str, db: StorageManager, top_k: int = 10) -> List[Dict[str, Any]]:
    """Intelligent multi-hop retrieval with context expansion."""
    query = normalize_query(query)
    weights = compute_dynamic_weights(query, db)

    # Step 1: Initial Retrieval (Pass dynamic weights)
    hop1_results = retrieve_no_event(query, db, top_k, weights=weights)

    # Step 2: Entity Extraction
    all_content = " ".join([r['content'] for r in hop1_results])
    entities = extract_entities(all_content)

    # Step 3: Build enriched query
    if entities:
        # Step 4: run second retrieval
        enriched_query = query + " " + " ".join(entities)
        hop2_results = retrieve_no_event(enriched_query, db, top_k, weights=weights)

        # Step 5: Merge results
        # Deduplicate by chunk_id
        seen_ids = {r['chunk_id'] for r in hop1_results}
        merged_results = list(hop1_results)
        for r in hop2_results:
            if r['chunk_id'] not in seen_ids:
                r['hop'] = 2
                merged_results.append(r)
                seen_ids.add(r['chunk_id'])

        # Deterministic ordering before final sort
        merged_results.sort(key=lambda x: x["chunk_id"])

        # Re-rank using scoring engine with stable tie-breaker
        merged_results.sort(
            key=lambda x: (
                -(x['score']['final'] if isinstance(x['score'], dict) else x['score']),
                x['chunk_id']
            )
        )
        top_results = merged_results[:top_k]
    else:
        top_results = hop1_results

    # Step 6: Context Expansion & Stitching
    final_results = []
    for r in top_results:
        # Expansion
        neighbors = expand_context(r, db, window_size=1)
        stitched_content = stitch_chunks(neighbors)

        final_results.append({
            "chunk_id": r["chunk_id"],
            "text": r.get("content", ""),
            "score": r["score"],
            "context": stitched_content,
            "source_path": r.get("path"),
            "hop": r.get("hop", 1)
        })

    # Final Deterministic Serialization Sort
    # Ensures consistent output ordering even if scores are identical
    final_results.sort(key=lambda x: (
        -(x['score']['final'] if isinstance(x['score'], dict) else x['score']),
        x['chunk_id']
    ))

    # 8. Feedback loop Integration
    retrieved_ids = [r['chunk_id'] for r in final_results]

    # TODO:
    # Capture and pass real user interaction signals (selected/dismissed) from the UI layer.
    # By default, we initialize with empty selections to prevent signal inflation.
    event = RetrievalEvent(query, retrieved_ids, selected_chunk_ids=[], dismissed_chunk_ids=[])
    log_event(event, db)

    return final_results

def retrieve_no_event(query: str, db: StorageManager, top_k: int = 10, weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """Ranked retrieval using the Scoring Engine without side-effects."""
    query = normalize_query(query)

    # 1. Deterministic Keywords extraction
    raw_keywords = re.findall(r'[A-Z][a-z0-9]+|\w{5,}', query)
    keywords = sorted(list(set(raw_keywords)))

    # 2. Deterministic Candidate Fetching
    candidates = []
    if keywords:
        # Use a stable sample of keywords
        for kw in keywords[:10]:
            chunk_batch = db.search_chunks_by_keyword(kw, limit=100)
            candidates.extend(chunk_batch)

    if not candidates:
        candidates = db.fetch_active_chunks()

    if not candidates:
        return []

    # Deduplicate and sort candidates deterministically before processing
    seen_cand = {}
    for c in candidates:
        seen_cand[c['id']] = c
    candidates = [seen_cand[cid] for cid in sorted(seen_cand.keys())]

    # 3. Embed query
    query_vector = embed_query(query)

    scorer = RetrievalScorer()
    results = []

    for chunk in candidates:
        # Convert BLOB back to vector
        chunk_vector = list(array.array('f', chunk['vector']))

        # 4. Compute similarity for the scorer
        similarity = cosine_similarity(query_vector, chunk_vector)

        # 5. Delegate scoring to the engine (Single source of truth)
        score_meta = {"raw_similarity": similarity}
        if weights:
            score_meta.update(weights)

        score_data = scorer.score(chunk, query_vector, metadata=score_meta)

        results.append({
            'chunk_id': chunk['id'],
            'content': chunk['content'],
            'score': {
                'final': score_data['final_score'],
                'components': score_data['components']
            },
            'path': chunk.get('path'),
            'document_id': chunk.get('document_id'),
            'start_line': chunk.get('start_line'),
            'end_line': chunk.get('end_line')
        })

    # 6. Ranking Pipeline: sort descending by final score with deterministic tie-breaker
    results.sort(key=lambda x: (-x['score']['final'], x['chunk_id']))
    return results[:top_k]

def expand_context(chunk: Dict[str, Any], db: StorageManager, window_size: int = 1) -> List[Dict[str, Any]]:
    """Fetch neighboring chunks from the SAME document."""
    doc_id = chunk.get('document_id')
    start_line = chunk.get('start_line')
    chunk_id = chunk.get('chunk_id') or chunk.get('id')

    if doc_id is None or start_line is None:
        return [chunk]

    neighbors = db.get_chunk_neighbors(doc_id, start_line, n=window_size, chunk_id=chunk_id)

    # Deterministic neighbor sorting
    all_chunks = neighbors + [chunk]
    all_chunks.sort(key=lambda x: (x.get('start_line', 0), x.get('id', '') or x.get('chunk_id', '')))

    return all_chunks

def stitch_chunks(chunks: List[Dict[str, Any]]) -> str:
    """Concatenate content in correct order, avoiding duplicates."""
    if not chunks:
        return ""

    # Ensure stable ordering before stitching
    chunks.sort(key=lambda x: (x.get('start_line', 0), x.get('id', '') or x.get('chunk_id', '')))

    stitched = []
    seen_content = set()
    for c in chunks:
        content = c['content']
        if content not in seen_content:
            stitched.append(content)
            seen_content.add(content)

    return "\n".join(stitched)

def extract_entities(text: str) -> List[str]:
    """Extract function names, class names, and identifiers deterministically."""
    entities = []
    # Extract function names: def NAME(
    entities.extend(re.findall(r'def\s+(\w+)\s*\(', text))
    # Extract class names: class NAME
    entities.extend(re.findall(r'class\s+(\w+)', text))
    # Extract capitalized tokens (identifiers)
    entities.extend(re.findall(r'\b[A-Z][a-zA-Z0-9_]+\b', text))

    # Enforce deterministic order and limit AFTER canonicalization
    sorted_entities = sorted(list(set(e.lower() for e in entities)))
    return sorted_entities[:10]
