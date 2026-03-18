import sqlite3
import json
from datetime import datetime

class StorageManager:
    def __init__(self, db_path: str = "achilles.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._enable_foreign_keys()
        self._enable_wal()
        self._create_schema()

    def _enable_foreign_keys(self):
        self.conn.execute("PRAGMA foreign_keys = ON;")

    def _enable_wal(self):
        self.conn.execute("PRAGMA journal_mode = WAL;")

    def _create_schema(self):
        cursor = self.conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE,
            content_hash TEXT,
            last_indexed TIMESTAMP
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            document_id INTEGER,
            content TEXT,
            content_hash TEXT,
            start_line INTEGER,
            end_line INTEGER,
            is_active BOOLEAN,
            created_at TIMESTAMP,
            FOREIGN KEY(document_id) REFERENCES documents(id)
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_id TEXT,
            vector BLOB,
            FOREIGN KEY(chunk_id) REFERENCES chunks(id)
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS retrieval_stats (
            chunk_id TEXT PRIMARY KEY,
            retrieval_count INTEGER DEFAULT 0,
            success_score REAL DEFAULT 1.0,
            last_accessed TIMESTAMP,
            FOREIGN KEY(chunk_id) REFERENCES chunks(id)
        );
        """)

        self.conn.commit()

    # ---------------- DOCUMENTS ----------------

    def upsert_document(self, path: str, content_hash: str) -> int:
        cursor = self.conn.cursor()
        now = datetime.utcnow()

        cursor.execute("""
        INSERT INTO documents (path, content_hash, last_indexed)
        VALUES (?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            content_hash=excluded.content_hash,
            last_indexed=excluded.last_indexed
        """, (path, content_hash, now))

        self.conn.commit()

        cursor.execute("SELECT id FROM documents WHERE path=?", (path,))
        return cursor.fetchone()["id"]

    def get_document_by_path(self, path: str):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE path=?", (path,))
        return cursor.fetchone()

    def get_all_documents(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents")
        return cursor.fetchall()

    # ---------------- CHUNKS ----------------

    def insert_chunk(self, chunk: dict):
        cursor = self.conn.cursor()

        cursor.execute("""
        INSERT OR REPLACE INTO chunks (
            id, document_id, content, content_hash,
            start_line, end_line, is_active, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk["id"],
            chunk["document_id"],
            chunk["content"],
            chunk["content_hash"],
            chunk["start_line"],
            chunk["end_line"],
            True,
            datetime.utcnow()
        ))

        self.conn.commit()

    def deactivate_chunks_for_document(self, document_id: int):
        cursor = self.conn.cursor()
        cursor.execute("""
        UPDATE chunks SET is_active=0 WHERE document_id=?
        """, (document_id,))
        self.conn.commit()

    def fetch_active_chunks(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM chunks WHERE is_active=1")
        return cursor.fetchall()

    # ---------------- EMBEDDINGS ----------------

    def insert_embedding(self, chunk_id: str, vector):
        cursor = self.conn.cursor()

        blob = json.dumps(vector)

        cursor.execute("""
        INSERT OR REPLACE INTO embeddings (chunk_id, vector)
        VALUES (?, ?)
        """, (chunk_id, blob))

        self.conn.commit()

    def get_embeddings(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM embeddings")
        return cursor.fetchall()

    # ---------------- RETRIEVAL STATS ----------------

    def update_retrieval_stats(self, chunk_id: str, used: bool):
        cursor = self.conn.cursor()
        now = datetime.utcnow()

        cursor.execute("SELECT * FROM retrieval_stats WHERE chunk_id=?", (chunk_id,))
        row = cursor.fetchone()

        if row:
            retrieval_count = row["retrieval_count"] + 1
            success_score = row["success_score"] + (1.0 if used else -0.1)
            success_score = max(success_score, 0.1)

            cursor.execute("""
            UPDATE retrieval_stats
            SET retrieval_count=?, success_score=?, last_accessed=?
            WHERE chunk_id=?
            """, (retrieval_count, success_score, now, chunk_id))
        else:
            cursor.execute("""
            INSERT INTO retrieval_stats (chunk_id, retrieval_count, success_score, last_accessed)
            VALUES (?, ?, ?, ?)
            """, (chunk_id, 1, 1.0 if used else 0.9, now))

        self.conn.commit()
