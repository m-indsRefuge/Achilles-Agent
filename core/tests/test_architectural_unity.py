import unittest
import os
import sys
import json
import sqlite3
import time

# Ensure core is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
core_path = os.path.join(os.path.dirname(current_dir), "core")
if os.path.exists(core_path):
    sys.path.append(core_path)

from scoring import RetrievalScorer
from storage import StorageManager
from retrieval import retrieve

class TestArchitecturalUnity(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_unity.sqlite"
        if os.path.exists(self.db_path): os.remove(self.db_path)
        self.db = StorageManager(self.db_path)

    def tearDown(self):
        self.db.close()
        if os.path.exists(self.db_path): os.remove(self.db_path)

    def test_single_source_of_truth_ranking(self):
        # Insert test data
        doc_id = self.db.upsert_document("test.py", "h")
        chunk_id = "chunk_unified"
        # Manually create embedding blob (384 zeros for MiniLM-L6)
        import array
        emb = array.array('f', [0.0]*384).tobytes()

        cursor = self.db.conn.cursor()
        cursor.execute("""
            INSERT INTO chunks (id, document_id, content, content_hash, start_line, end_line, is_active)
            VALUES (?, ?, ?, ?, ?, ?, 1)
        """, (chunk_id, doc_id, "unified content", "ch", 1, 1))
        cursor.execute("INSERT INTO embeddings (chunk_id, vector) VALUES (?, ?)", (chunk_id, emb))
        cursor.execute("INSERT INTO retrieval_stats (chunk_id, success_score) VALUES (?, ?)", (chunk_id, 10.0))
        self.db.conn.commit()

        # Retrieve
        results = retrieve("unified", self.db, top_k=1)

        # Verify result contains explainable scoring components
        res = results[0]
        self.assertIn("score", res)
        self.assertIsInstance(res["score"], dict)
        self.assertIn("final", res["score"])
        self.assertIn("components", res["score"])
        self.assertIn("feedback", res["score"]["components"])

    def test_no_legacy_fields(self):
        # Confirm usage_score is gone from storage and logic
        from storage import StorageManager
        import inspect

        # Check storage schema (programmatic check if possible, or just method inspection)
        methods = [m[0] for m in inspect.getmembers(self.db, predicate=inspect.ismethod)]
        # We don't want any methods referencing usage_score
        self.assertFalse(any("usage_score" in m for m in methods))

        # Check SQLite columns
        cursor = self.db.conn.cursor()
        cursor.execute("PRAGMA table_info(retrieval_stats)")
        columns = [c[1] for c in cursor.fetchall()]
        self.assertNotIn("usage_score", columns)
        self.assertIn("success_score", columns)

if __name__ == "__main__":
    unittest.main()
