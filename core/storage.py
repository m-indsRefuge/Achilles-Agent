import sqlite3
import os
import time
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
                selected_chunk_ids TEXT
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

    def update_retrieval_stats(self, chunk_id: str, used: bool):
        MAX_SUCCESS_SCORE = 50.0

        with self.lock:
            cursor = self.conn.cursor()

            # Fetch current state
            cursor.execute("SELECT success_score FROM retrieval_stats WHERE chunk_id = ?", (chunk_id,))
            row = cursor.fetchone()
            if not row:
                return

            current_score = row[0]
            new_score = current_score

            # Only allow positive reinforcement
            if used:
                new_score += 1.0

            # Apply Hard Cap
            new_score = min(new_score, MAX_SUCCESS_SCORE)

            # Persist
            cursor.execute("""
                UPDATE retrieval_stats
                SET retrieval_count = retrieval_count + 1,
                    success_score = ?,
                    last_accessed = CURRENT_TIMESTAMP,
                    last_updated = CURRENT_TIMESTAMP
                WHERE chunk_id = ?
            """, (new_score, chunk_id))
            self.conn.commit()

    def insert_retrieval_event(self, query: str, retrieved_ids: List[str], selected_ids: List[str]):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO retrieval_events (query, retrieved_chunk_ids, selected_chunk_ids)
                VALUES (?, ?, ?)
            """, (query, json.dumps(retrieved_ids), json.dumps(selected_ids)))
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

    def get_chunk_neighbors(self, doc_id: int, start_line: int, n: int = 1) -> List[Dict[str, Any]]:
        """Fetch neighboring chunks from the same document."""
        cursor = self.conn.cursor()

        # Fetch previous N
        cursor.execute("""
            SELECT c.id, c.content, c.document_id, e.vector, s.retrieval_count, s.success_score, s.last_accessed, c.created_at, d.path, c.start_line, c.end_line
            FROM chunks c
            JOIN embeddings e ON c.id = e.chunk_id
            JOIN retrieval_stats s ON c.id = s.chunk_id
            JOIN documents d ON c.document_id = d.id
            WHERE c.document_id = ? AND c.is_active = 1
            AND c.start_line < ?
            ORDER BY c.start_line DESC LIMIT ?
        """, (doc_id, start_line, n))
        prev_chunks = cursor.fetchall()

        # Fetch next N
        cursor.execute("""
            SELECT c.id, c.content, c.document_id, e.vector, s.retrieval_count, s.success_score, s.last_accessed, c.created_at, d.path, c.start_line, c.end_line
            FROM chunks c
            JOIN embeddings e ON c.id = e.chunk_id
            JOIN retrieval_stats s ON c.id = s.chunk_id
            JOIN documents d ON c.document_id = d.id
            WHERE c.document_id = ? AND c.is_active = 1
            AND c.start_line > ?
            ORDER BY c.start_line ASC LIMIT ?
        """, (doc_id, start_line, n))
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
