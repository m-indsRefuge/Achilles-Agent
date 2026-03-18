import sqlite3
import json
import os
from typing import List, Dict, Any

class STMStore:
    """
    SQLite-based storage for Short-Term Memory.
    Ensures robust persistence and concurrency for chat history.
    """
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        self.conn = None
        self._connect()

    def _connect(self):
        self.conn = sqlite3.connect(self.storage_path, check_same_thread=False)
        self._setup_db()

    def _setup_db(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def add(self, entry: Dict[str, Any], max_size: int = 50):
        if not self.conn: self._connect()
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO memory (data) VALUES (?)", (json.dumps(entry),))

        # Enforce max size
        cursor.execute("SELECT COUNT(*) FROM memory")
        count = cursor.fetchone()[0]
        if count > max_size:
            cursor.execute("""
                DELETE FROM memory WHERE id IN (
                    SELECT id FROM memory ORDER BY timestamp ASC LIMIT ?
                )
            """, (count - max_size,))

        self.conn.commit()

    def get_all(self) -> List[Dict[str, Any]]:
        if not self.conn: return []
        cursor = self.conn.cursor()
        cursor.execute("SELECT data FROM memory ORDER BY timestamp ASC")
        rows = cursor.fetchall()
        return [json.loads(row[0]) for row in rows]

    def clear(self):
        if self.conn:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM memory")
            self.conn.commit()
            self.conn.close()
            self.conn = None
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
