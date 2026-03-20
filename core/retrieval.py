import math
import time
import array
import re
from typing import List, Dict, Any
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

def retrieve(query: str, db: StorageManager, top_k: int = 10) -> List[Dict[str, Any]]:
    """Intelligent multi-hop retrieval with context expansion."""
    # Step 1: Initial Retrieval
    hop1_results = retrieve_no_event(query, db, top_k)

    # Step 2: Entity Extraction
    all_content = " ".join([r['content'] for r in hop1_results])
    entities = extract_entities(all_content)

    # Step 3: Build enriched query
    if entities:
        enriched_query = query + " " + " ".join(entities[:10]) # Limit entities
        # Step 4: run second retrieval
        hop2_results = retrieve_no_event(enriched_query, db, top_k)

        # Step 5: Merge results
        # Deduplicate by chunk_id
        seen_ids = {r['chunk_id'] for r in hop1_results}
        merged_results = list(hop1_results)
        for r in hop2_results:
            if r['chunk_id'] not in seen_ids:
                r['hop'] = 2
                merged_results.append(r)
                seen_ids.add(r['chunk_id'])

        # Re-rank using scoring engine (sorting by score.final)
        merged_results.sort(key=lambda x: x['score']['final'] if isinstance(x['score'], dict) else x['score'], reverse=True)
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

    # 8. Feedback loop Integration
    retrieved_ids = [r['chunk_id'] for r in final_results]

    # TODO:
    # Replace implicit selection with real user interaction signal
    # (e.g. clicked result, accepted suggestion, etc.)
    # For now, treat all retrieved results as "implicitly useful" to prevent incorrect penalties.
    event = RetrievalEvent(query, retrieved_ids, retrieved_ids)
    log_event(event, db)

    return final_results

def retrieve_no_event(query: str, db: StorageManager, top_k: int = 10) -> List[Dict[str, Any]]:
    """Ranked retrieval using the Scoring Engine without side-effects."""
    # 1. Keywords extraction for pre-filtering
    # Heuristic: prioritized words (capitalized or specific patterns)
    keywords = re.findall(r'[A-Z][a-z0-9]+|\w{5,}', query)

    # 2. Fetch candidates (Filtered or Full Scan)
    candidates = []
    if keywords:
        # Use more selective keywords first
        sorted_keywords = sorted(keywords, key=len, reverse=True)
        for kw in sorted_keywords[:5]:
            chunk_batch = db.search_chunks_by_keyword(kw, limit=50)
            candidates.extend(chunk_batch)
            if len(candidates) > 200: # Threshold for sufficient candidates
                break

    if not candidates:
        # Fallback to a broader but still somewhat limited scan if possible,
        # or full scan if necessary for small repos.
        candidates = db.fetch_active_chunks()

    if not candidates:
        return []

    # Deduplicate candidates
    seen_cand = {}
    for c in candidates:
        seen_cand[c['id']] = c
    candidates = list(seen_cand.values())

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
        score_data = scorer.score(chunk, query_vector, metadata={"raw_similarity": similarity})

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

    # 6. Ranking Pipeline: sort descending by final score
    results.sort(key=lambda x: x['score']['final'], reverse=True)
    return results[:top_k]

def expand_context(chunk: Dict[str, Any], db: StorageManager, window_size: int = 1) -> List[Dict[str, Any]]:
    """Fetch neighboring chunks from the SAME document."""
    doc_id = chunk.get('document_id')
    start_line = chunk.get('start_line')
    end_line = chunk.get('end_line')

    if doc_id is None or start_line is None or end_line is None:
        return [chunk]

    neighbors = db.get_chunk_neighbors(doc_id, start_line, n=window_size)

    # Combine and sort to ensure order
    # The get_chunk_neighbors returns prev and next. We insert original in middle.
    all_chunks = neighbors + [chunk]
    all_chunks.sort(key=lambda x: x.get('start_line', 0))

    return all_chunks

def stitch_chunks(chunks: List[Dict[str, Any]]) -> str:
    """Concatenate content in correct order, avoiding duplicates."""
    if not chunks:
        return ""

    stitched = []
    seen_content = set()
    for c in chunks:
        content = c['content']
        if content not in seen_content:
            stitched.append(content)
            seen_content.add(content)

    return "\n".join(stitched)

def extract_entities(text: str) -> List[str]:
    """Extract function names, class names, and identifiers."""
    entities = []
    # Extract function names: def NAME(
    entities.extend(re.findall(r'def\s+(\w+)\s*\(', text))
    # Extract class names: class NAME
    entities.extend(re.findall(r'class\s+(\w+)', text))
    # Extract capitalized tokens (identifiers)
    entities.extend(re.findall(r'\b[A-Z][a-zA-Z0-9_]+\b', text))

    return list(set(entities))
