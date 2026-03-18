import sqlite3
import os
import time
from typing import List, Dict, Any, Optional

class StorageManager:
    """
    SQLite-based single source of truth for the memory engine.
    Handles schema exactly as requested: documents, chunks, embeddings, and retrieval statistics.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
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
                FOREIGN KEY(chunk_id) REFERENCES chunks(id)
            )
        """)

        # Indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_active ON chunks(is_active)")

        self.conn.commit()

    def upsert_document(self, path: str, content_hash: str) -> int:
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
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO chunks (id, document_id, content, content_hash, start_line, end_line, is_active)
            VALUES (?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(id) DO UPDATE SET is_active=1
        """, (
            chunk['id'], chunk['document_id'], chunk['content'],
            chunk['content_hash'], chunk['start_line'], chunk['end_line']
        ))

        # Initialize stats if not present
        cursor.execute("INSERT OR IGNORE INTO retrieval_stats (chunk_id) VALUES (?)", (chunk['id'],))
        self.conn.commit()

    def deactivate_chunks_for_document(self, document_id: int):
        cursor = self.conn.cursor()
        cursor.execute("UPDATE chunks SET is_active = 0 WHERE document_id = ?", (document_id,))
        self.conn.commit()

    def insert_embedding(self, chunk_id: str, vector_blob: bytes):
        cursor = self.conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO embeddings (chunk_id, vector) VALUES (?, ?)", (chunk_id, vector_blob))
        self.conn.commit()

    def fetch_active_chunks(self) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT c.id, c.content, c.document_id, e.vector, s.retrieval_count, s.success_score, s.last_accessed, c.created_at
            FROM chunks c
            JOIN embeddings e ON c.id = e.chunk_id
            JOIN retrieval_stats s ON c.id = s.chunk_id
            WHERE c.is_active = 1
        """)
        rows = cursor.fetchall()
        return [
            {
                "id": r[0], "content": r[1], "document_id": r[2], "vector": r[3],
                "retrieval_count": r[4], "success_score": r[5], "last_accessed": r[6], "created_at": r[7]
            } for r in rows
        ]

    def update_retrieval_stats(self, chunk_id: str, used: bool):
        cursor = self.conn.cursor()
        score_delta = 1.0 if used else -0.1
        cursor.execute("""
            UPDATE retrieval_stats
            SET retrieval_count = retrieval_count + 1,
                success_score = MAX(0.1, success_score + ?),
                last_accessed = CURRENT_TIMESTAMP
            WHERE chunk_id = ?
        """, (score_delta, chunk_id))
        self.conn.commit()

    def close(self):
        self.conn.close()
