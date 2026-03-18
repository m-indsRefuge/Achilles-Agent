import sqlite3
import json
import os
import numpy as np
from typing import List, Dict, Any, Optional

class CognitiveMemoryStore:
    """
    Unified relational store for the Cognitive Memory Engine.
    Handles documents, chunks, and retrieval performance tracking.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._setup_schema()

    def _setup_schema(self):
        cursor = self.conn.cursor()
        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE,
                file_hash TEXT,
                last_indexed_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                doc_id INTEGER,
                content_hash TEXT,
                text_content TEXT,
                sequence_index INTEGER,
                embedding BLOB,
                FOREIGN KEY(doc_id) REFERENCES documents(id)
            )
        """)
        # Retrieval Stats table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_stats (
                chunk_id TEXT PRIMARY KEY,
                retrieval_count INTEGER DEFAULT 0,
                success_score FLOAT DEFAULT 0.0,
                last_retrieved_at DATETIME,
                FOREIGN KEY(chunk_id) REFERENCES chunks(id)
            )
        """)
        # Indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash)")
        self.conn.commit()

    def add_document(self, path: str, file_hash: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO documents (path, file_hash, last_indexed_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
            (path, file_hash)
        )
        self.conn.commit()
        return cursor.lastrowid

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        cursor = self.conn.cursor()
        for c in chunks:
            # Store embedding as BLOB
            emb_blob = np.array(c['embedding'], dtype='float32').tobytes() if 'embedding' in c else None
            cursor.execute("""
                INSERT OR REPLACE INTO chunks (id, doc_id, content_hash, text_content, sequence_index, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (c['id'], c['doc_id'], c['content_hash'], c['text_content'], c['sequence_index'], emb_blob))

            # Initialize stats
            cursor.execute("INSERT OR IGNORE INTO retrieval_stats (chunk_id) VALUES (?)", (c['id'],))
        self.conn.commit()

    def record_retrieval(self, chunk_id: str, success: bool = False):
        cursor = self.conn.cursor()
        score_inc = 1.0 if success else 0.0
        cursor.execute("""
            UPDATE retrieval_stats
            SET retrieval_count = retrieval_count + 1,
                success_score = success_score + ?,
                last_retrieved_at = CURRENT_TIMESTAMP
            WHERE chunk_id = ?
        """, (score_inc, chunk_id))
        self.conn.commit()

    def get_neighbors(self, doc_id: int, sequence_index: int, n: int = 1) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, text_content, sequence_index FROM chunks
            WHERE doc_id = ? AND sequence_index BETWEEN ? AND ?
            ORDER BY sequence_index ASC
        """, (doc_id, sequence_index - n, sequence_index + n))
        rows = cursor.fetchall()
        return [{"id": r[0], "text_content": r[1], "sequence_index": r[2]} for r in rows]

    def close(self):
        self.conn.close()
