# Achilles Agent — Advanced Memory & Retrieval Architecture

## 1. Deterministic, Repeatable Indexing Engine

A deterministic indexing engine ensures that the same input always produces the same indexed state.

### Core Principles
- Content hashing (SHA256 per file + per chunk)
- Idempotent operations (safe to re-run)
- Incremental updates (only changed files processed)
- Stable chunking strategy

### Pipeline
1. Scan repository
2. Normalize paths (absolute, canonical)
3. Compute file hash
4. Compare against stored hash
5. If changed:
   - Chunk file (AST-aware for code, semantic for text)
   - Generate embeddings
   - Assign chunk_id = hash(file_path + chunk_index + content_hash)
6. Store in DB
7. Mark stale chunks as inactive

### Key Insight
Indexing is not embedding — it is **state synchronization between filesystem and memory**.

---

## 2. SQLite Storage Layer

### Why SQLite
- Zero-config
- ACID compliant
- Fast enough for local systems

### Schema

#### documents
- id (PK)
- path (unique)
- hash
- last_indexed

#### chunks
- id (PK)
- document_id (FK)
- content
- content_hash
- start_line
- end_line
- created_at
- is_active

#### embeddings
- chunk_id (FK)
- vector (BLOB or JSON)

#### retrieval_stats
- chunk_id
- retrieval_count
- success_score
- last_accessed

### Indexes
- index on path
- index on content_hash
- index on retrieval_count

---

## 3. Recency Weighting

Score = similarity * alpha + recency_decay * beta

Where:
recency_decay = exp(-lambda * age_seconds)

This biases newer or recently used knowledge.

---

## 4. Context Windows

After retrieval:
- Expand chunk ± N neighboring chunks
- Merge into coherent context block
- Respect token limits

This prevents fragmented understanding.

---

## 5. Task-Aware Retrieval

Classify query into:
- debugging
- architecture
- factual lookup

Apply filters:
- file types
- directories
- recency thresholds

---

## 6. Multi-Hop Recall

Step 1: initial retrieval
Step 2: extract entities (functions, classes)
Step 3: re-query using extracted entities
Step 4: merge results

This enables reasoning across files.

---

## 7. Feedback Loop (Usage → Training)

Track:
- retrieved chunks
- selected chunks
- user corrections

Store signals in retrieval_stats.

---

## 8. Reinforcement from Successful Retrievals

Increase score:
- if chunk used in final answer
- if user accepts output

Decrease score:
- if ignored

Use as ranking multiplier.

---

## 9. Memory Pruning / Refinement

Strategies:
- remove stale chunks (low score + old)
- merge duplicate chunks
- re-chunk large files
- compress rarely used knowledge

---

## Final Insight

You are not building a chatbot.
You are building a **self-improving knowledge system**.
