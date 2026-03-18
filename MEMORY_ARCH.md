# Achilles Cognitive Memory Engine: Technical Architecture

This document defines the high-level and deeply technical architecture of the Achilles Cognitive Memory Engine, a self-improving, deterministic system for long-term agentic memory.

---

## 1. Deterministic, Repeatable Indexing Engine

The indexing engine ensures that identical inputs always result in identical indexed outputs, enabling idempotent re-indexing and efficient incremental updates.

### Key Components:
- **Content Hashing**: Uses SHA-256 for both file-level (detecting changes) and chunk-level (detecting identical content across files).
- **Incremental Pipeline**: `scan` → `file_hash` → `diff` (against existing docs) → `AST-aware chunking` → `embedding` → `store` → `cleanup`.
- **Stable Chunk IDs**: Defined as `H(file_path + chunk_index + content_hash)`. This allows tracking a chunk even if its position in the file changes slightly.
- **AST-Aware Chunking**: For code, it uses tree-sitter or similar to preserve logical blocks (functions, classes). For text, it uses semantic boundaries (paragraphs, sections).

### Invariants:
- If `hash(file_content)` is unchanged, no re-processing occurs.
- If a file is deleted, all chunks associated with its `file_path` that aren't shared (via `content_hash`) are invalidated.

---

## 2. SQLite-Based Storage Layer

A unified relational store for identity, metadata, and vectors.

### Schema Design:
- **`documents`**: `id`, `path`, `file_hash`, `last_indexed_at`, `total_chunks`.
- **`chunks`**: `id` (stable), `doc_id`, `content_hash`, `text_content`, `sequence_index`, `byte_offset`, `tokens`.
- **`embeddings`**: `chunk_id`, `vector` (BLOB), `model_name`.
- **`retrieval_stats`**: `chunk_id`, `last_retrieved_at`, `retrieval_count`, `success_score`.

### Implementation Details:
- **Embeddings Storage**: Stored as `float32` BLOBs for maximum performance. Vector search is accelerated via an external index (FAISS) which is kept in sync with the SQLite source of truth.
- **Concurrency**: Uses Write-Ahead Logging (WAL) mode to support concurrent reads during indexing.

---

## 3. Recency Weighting

Retrieval scores are adjusted to prioritize fresh knowledge while retaining deep context.

### Mathematical Formulation:
`FinalScore = SemanticSimilarity * DecayFactor + BoostFactor`
`DecayFactor = exp(-λ * Δt)`
- `Δt`: Time since last document update or last retrieval.
- `λ`: Tunable decay constant.

This prevents stale documentation or deprecated code from dominating search results just because of high semantic overlap.

---

## 4. Context Windows

Instead of returning isolated chunks, the engine expands results into coherent context blocks.

### Expansion Strategy:
- **Neighbor Expansion**: For every hit, retrieve chunks at `sequence_index ± N`.
- **Logical Boundary Check**: Expansion continues until a logical boundary (end of function/class) is reached or the token limit is hit.
- **Benefits**: Reduces fragmentation and provides the LLM with enough local context to understand variable definitions or class hierarchies.

---

## 5. Task-Aware Retrieval

Retrieval parameters adapt dynamically based on the agent's current intent.

### Strategy:
- **Intent Classification**: Classifies query into `Lookup` (strict semantic), `Architecture` (high-level expansion), `Debugging` (heavy diagnostics/logs), or `Explanation`.
- **Dynamic Filtering**:
    - `Debugging` intent filters for `.log`, `.test`, and recent commits.
    - `Architecture` intent filters for entry points, config files, and high-level class definitions.
- **Scoring Adjustment**: Increases weights for specific file types or directories (e.g., `src/` vs `tests/`) based on task context.

---

## 6. Multi-Hop Recall

Enables cross-file reasoning by performing iterative retrieval.

### Workflow:
1. **Initial Retrieval**: Fetch chunks related to the primary query.
2. **Symbol Extraction**: Parse results to identify mentioned entities (e.g., `UserService`, `API_ENDPOINT`).
3. **Recursive Search**: Automatically perform follow-up searches for the extracted symbols.
4. **Context Merge**: De-duplicate and rank the union of all results to build the final context set.

---

## 7. Feedback Loop (Usage → Training)

A dedicated system to capture and store signals for system optimization.

### Logging Data Structure:
- **`event_id`**: Unique ID.
- **`query`**: The raw user/agent query.
- **`retrieved_chunks`**: List of chunk IDs returned.
- **`used_chunks`**: Sub-set of IDs actually included in the final LLM prompt.
- **`user_feedback`**: Explicit (thumb up/down) or implicit (edit of the suggestion).

---

## 8. Reinforcement from Successful Retrievals

Ranking behavior evolves based on historical success.

### Mechanism:
- **`success_score` Update**: When a chunk is marked as "used" or receives positive feedback, its `success_score` in `retrieval_stats` is incremented.
- **Ranking Integration**: `RankingScore = FinalScore * (1 + log(1 + success_score))`.
- **Long-term Behavior**: Frequently helpful code snippets (e.g., core utilities) bubble to the top, while noisy or misleading chunks are naturally penalized.

---

## 9. Memory Pruning and Refinement

Maintains high-quality memory by purging low-utility data.

### Strategies:
- **Stale Chunk Invalidation**: Chunks with `retrieval_count == 0` after `X` months are candidates for deletion.
- **Semantic Deduplication**: Chunks with >98% similarity within the same project are merged.
- **Summarization**: Rarely used but historically relevant knowledge is passed through an LLM to generate a dense summary, replacing the raw chunks to save space.
- **Re-chunking**: If feedback indicates a chunk is often "cut off," the engine marks that region for re-indexing with adjusted boundaries.

---

## Conclusion

The Achilles Cognitive Memory Engine is a **self-improving cognitive memory engine**. It moves beyond passive vector search into active, task-aware context synthesis. By combining deterministic indexing with reinforcement-driven ranking, it creates a persistent, evolving understanding of the codebase that grows more precise with every interaction.
