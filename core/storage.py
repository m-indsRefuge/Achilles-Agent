import sqlite3
import os
import time
import math
import threading
import json
from typing import List, Dict, Any, Optional

class StorageManager:
    """
    SQLite-based single source of truth for the memory engine.
    Handles schema exactly as requested: documents, chunks, embeddings, and retrieval statistics.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = threading.Lock()
        parent_dir = os.path.dirname(self.db_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self._create_schema()

    def _create_schema(self):
        cursor = self.conn.cursor()

        # 1. Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE,
                content_hash TEXT,
                last_indexed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 2. Chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id INTEGER,
                content TEXT,
                content_hash TEXT,
                start_line INTEGER,
                end_line INTEGER,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(document_id) REFERENCES documents(id)
            )
        """)

        # 3. Embeddings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id TEXT,
                vector BLOB,
                FOREIGN KEY(chunk_id) REFERENCES chunks(id)
            )
        """)

        # 4. Retrieval Stats table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_stats (
                chunk_id TEXT PRIMARY KEY,
                retrieval_count INTEGER DEFAULT 0,
                success_score REAL DEFAULT 1.0,
                last_accessed TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(chunk_id) REFERENCES chunks(id)
            )
        """)

        # 5. Retrieval Events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                retrieved_chunk_ids TEXT,
                selected_chunk_ids TEXT,
                dismissed_chunk_ids TEXT
            )
        """)

        # Indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_active ON chunks(is_active)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_content ON chunks(content)")

        self.conn.commit()

    def upsert_document(self, path: str, content_hash: str) -> int:
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO documents (path, content_hash, last_indexed)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(path) DO UPDATE SET
                    content_hash=excluded.content_hash,
                    last_indexed=excluded.last_indexed
            """, (path, content_hash))
            self.conn.commit()

            cursor.execute("SELECT id FROM documents WHERE path = ?", (path,))
            return cursor.fetchone()[0]

    def get_document_by_path(self, path: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, path, content_hash FROM documents WHERE path = ?", (path,))
        row = cursor.fetchone()
        if row:
            return {"id": row[0], "path": row[1], "content_hash": row[2]}
        return None

    def insert_chunk(self, chunk: Dict[str, Any]):
        self.insert_chunks([chunk])

    def insert_chunks(self, chunks: List[Dict[str, Any]]):
        with self.lock:
            cursor = self.conn.cursor()
            try:
                for chunk in chunks:
                    cursor.execute("""
                        INSERT INTO chunks (id, document_id, content, content_hash, start_line, end_line, is_active)
                        VALUES (?, ?, ?, ?, ?, ?, 1)
                        ON CONFLICT(id) DO UPDATE SET is_active=1
                    """, (
                        chunk['id'], chunk['document_id'], chunk['content'],
                        chunk['content_hash'], chunk['start_line'], chunk['end_line']
                    ))
                    cursor.execute("INSERT OR IGNORE INTO retrieval_stats (chunk_id) VALUES (?)", (chunk['id'],))
                self.conn.commit()
            except sqlite3.Error as e:
                self.conn.rollback()
                raise e

    def deactivate_chunks_for_document(self, document_id: int):
        with self.lock:
            cursor = self.conn.cursor()
            try:
                cursor.execute("UPDATE chunks SET is_active = 0 WHERE document_id = ?", (document_id,))
                self.conn.commit()
            except sqlite3.Error as e:
                self.conn.rollback()
                raise e

    def insert_embedding(self, chunk_id: str, vector_blob: bytes):
        self.insert_embeddings([(chunk_id, vector_blob)])

    def insert_embeddings(self, embeddings: List[tuple]):
        with self.lock:
            cursor = self.conn.cursor()
            try:
                cursor.executemany("INSERT OR REPLACE INTO embeddings (chunk_id, vector) VALUES (?, ?)", embeddings)
                self.conn.commit()
            except sqlite3.Error as e:
                self.conn.rollback()
                raise e

    def fetch_active_chunks(self) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT c.id, c.content, c.document_id, e.vector, s.retrieval_count, s.success_score, s.last_accessed, c.created_at, d.path, c.start_line, c.end_line, s.last_updated
            FROM chunks c
            JOIN embeddings e ON c.id = e.chunk_id
            JOIN retrieval_stats s ON c.id = s.chunk_id
            JOIN documents d ON c.document_id = d.id
            WHERE c.is_active = 1
        """)
        return self._map_rows_to_chunks(cursor.fetchall(), include_lines=True)

    def update_retrieval_stats(self, chunk_id: str, signal: str = "neutral", weight: float = 1.0):
        """
        Updates retrieval statistics with Stability and Convergence logic.
        - Includes Temporal Decay
        - Delta Capping
        - Momentum Smoothing
        - Min/Max Bounding
        - Local Competition (Relative Suppression)
        """
        MAX_SUCCESS_SCORE = 50.0
        MIN_SUCCESS_SCORE = 0.1
        DECAY_LAMBDA = 1e-6
        MAX_DELTA = 2.0
        ALPHA = 0.3 # Momentum factor
        SUPPRESSION_FACTOR = 0.01 # Local competition multiplier

        with self.lock:
            cursor = self.conn.cursor()

            # Fetch current state
            cursor.execute("SELECT success_score, last_updated FROM retrieval_stats WHERE chunk_id = ?", (chunk_id,))
            row = cursor.fetchone()
            if not row:
                return

            old_score, last_updated_str = row
            current_time = time.time()

            # 1. Apply Temporal Decay
            last_updated = current_time
            if last_updated_str:
                try:
                    import datetime
                    if 'T' in last_updated_str:
                        last_updated = datetime.datetime.fromisoformat(last_updated_str).timestamp()
                    else:
                        last_updated = datetime.datetime.strptime(last_updated_str, '%Y-%m-%d %H:%M:%S').timestamp()
                except:
                    last_updated = current_time

            age = max(0, current_time - last_updated)
            decay = math.exp(-DECAY_LAMBDA * age)
            base_score = old_score * decay

            # 2. Local Competition (Relative Suppression)
            # If not selected, apply light suppression to base score
            if signal != "selected":
                base_score *= (1 - SUPPRESSION_FACTOR)

            # 3. Calculate Raw Delta with Capping
            raw_delta = 0.0
            if signal == "selected":
                raw_delta = 1.0 * weight
            elif signal == "dismissed":
                raw_delta = -0.2 * weight

            capped_delta = max(-MAX_DELTA, min(MAX_DELTA, raw_delta))

            # 4. Momentum Smoothing
            # new_score = alpha * (base + delta) + (1 - alpha) * old
            target_score = base_score + capped_delta
            new_score = (ALPHA * target_score) + ((1 - ALPHA) * old_score)

            # 5. Apply Hard Bounds
            new_score = max(MIN_SUCCESS_SCORE, min(new_score, MAX_SUCCESS_SCORE))

            # 5. Persist
            cursor.execute("""
                UPDATE retrieval_stats
                SET retrieval_count = retrieval_count + 1,
                    success_score = ?,
                    last_accessed = CURRENT_TIMESTAMP,
                    last_updated = CURRENT_TIMESTAMP
                WHERE chunk_id = ?
            """, (new_score, chunk_id))
            self.conn.commit()

    def insert_retrieval_event(self, query: str, retrieved_ids: List[str], selected_ids: List[str], dismissed_ids: Optional[List[str]] = None):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO retrieval_events (query, retrieved_chunk_ids, selected_chunk_ids, dismissed_chunk_ids)
                VALUES (?, ?, ?, ?)
            """, (query, json.dumps(retrieved_ids), json.dumps(selected_ids), json.dumps(dismissed_ids or [])))
            self.conn.commit()

    def get_query_frequency(self, query: str) -> int:
        """Count how many times a query has been performed historically."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM retrieval_events WHERE query = ?", (query,))
        return cursor.fetchone()[0]

    def normalize_retrieval_set_scores(self, chunk_ids: List[str]):
        """
        Local normalization: Scales all scores in the retrieval set relative to the maximum score (0->1 range).
        Preserves relative differences and respects the floor constraint.
        """
        if not chunk_ids:
            return

        MIN_SUCCESS_SCORE = 0.1

        with self.lock:
            cursor = self.conn.cursor()

            # 1. Fetch current scores
            placeholders = ','.join(['?'] * len(chunk_ids))
            cursor.execute(f"SELECT chunk_id, success_score FROM retrieval_stats WHERE chunk_id IN ({placeholders})", chunk_ids)
            results = cursor.fetchall()

            if not results:
                return

            # 2. Find maximum
            max_score = max(r[1] for r in results)

            # 3. Apply normalization
            if max_score > 0:
                for chunk_id, current_score in results:
                    normalized_score = current_score / max_score
                    # Maintain floor
                    final_score = max(MIN_SUCCESS_SCORE, normalized_score)

                    cursor.execute("""
                        UPDATE retrieval_stats
                        SET success_score = ?
                        WHERE chunk_id = ?
                    """, (final_score, chunk_id))

            self.conn.commit()

    def search_chunks_by_keyword(self, keyword: str, limit: int = 200) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT c.id, c.content, c.document_id, e.vector, s.retrieval_count, s.success_score, s.last_accessed, c.created_at, d.path, c.start_line, c.end_line, s.last_updated
            FROM chunks c
            JOIN embeddings e ON c.id = e.chunk_id
            JOIN retrieval_stats s ON c.id = s.chunk_id
            JOIN documents d ON c.document_id = d.id
            WHERE c.is_active = 1 AND c.content LIKE ?
            LIMIT ?
        """, (f"%{keyword}%", limit))
        rows = cursor.fetchall()
        return self._map_rows_to_chunks(rows, include_lines=True)

    def get_chunk_neighbors(self, doc_id: int, start_line: int, n: int = 1, chunk_id: str = None) -> List[Dict[str, Any]]:
        """Fetch neighboring chunks from the same document."""
        cursor = self.conn.cursor()

        # Fetch previous N
        cursor.execute("""
            SELECT c.id, c.content, c.document_id, e.vector, s.retrieval_count, s.success_score, s.last_accessed, c.created_at, d.path, c.start_line, c.end_line, s.last_updated
            FROM chunks c
            LEFT JOIN embeddings e ON c.id = e.chunk_id
            LEFT JOIN retrieval_stats s ON c.id = s.chunk_id
            JOIN documents d ON c.document_id = d.id
            WHERE c.document_id = ? AND c.is_active = 1
            AND (c.start_line < ? OR (c.start_line = ? AND c.id < ?))
            ORDER BY c.start_line DESC, c.id DESC LIMIT ?
        """, (doc_id, start_line, start_line, chunk_id or "", n))
        prev_chunks = cursor.fetchall()

        # Fetch next N
        cursor.execute("""
            SELECT c.id, c.content, c.document_id, e.vector, s.retrieval_count, s.success_score, s.last_accessed, c.created_at, d.path, c.start_line, c.end_line, s.last_updated
            FROM chunks c
            LEFT JOIN embeddings e ON c.id = e.chunk_id
            LEFT JOIN retrieval_stats s ON c.id = s.chunk_id
            JOIN documents d ON c.document_id = d.id
            WHERE c.document_id = ? AND c.is_active = 1
            AND (c.start_line > ? OR (c.start_line = ? AND c.id > ?))
            ORDER BY c.start_line ASC, c.id ASC LIMIT ?
        """, (doc_id, start_line, start_line, chunk_id or "", n))
        next_chunks = cursor.fetchall()

        rows = prev_chunks[::-1] + next_chunks
        return self._map_rows_to_chunks(rows, include_lines=True)

    def get_top_chunks(self, limit: int = 10) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT chunk_id, retrieval_count, success_score
            FROM retrieval_stats
            ORDER BY success_score DESC, retrieval_count DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        return [{"chunk_id": r[0], "retrieval_count": r[1], "success_score": r[2]} for r in rows]

    def _map_rows_to_chunks(self, rows, include_lines=False) -> List[Dict[str, Any]]:
        chunks = []
        for r in rows:
            chunk = {
                "id": r[0], "content": r[1], "document_id": r[2], "vector": r[3],
                "retrieval_count": r[4], "success_score": r[5], "last_accessed": r[6], "created_at": r[7],
                "path": r[8]
            }
            if include_lines:
                chunk["start_line"] = r[9]
                chunk["end_line"] = r[10]
                if len(r) > 11:
                    chunk["last_updated"] = r[11]
            chunks.append(chunk)
        return chunks

    def close(self):
        self.conn.close()
