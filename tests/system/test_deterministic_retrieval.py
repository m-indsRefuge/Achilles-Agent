import unittest
import os
import array
from core.storage import StorageManager
from core.retrieval import retrieve

class TestDeterministicRetrieval(unittest.TestCase):
    def setUp(self):
        self.db_path = "backend/memory_layer/storage/determinism_test.sqlite"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.db = StorageManager(self.db_path)

        # Setup test data
        self.dummy_vector = array.array('f', [0.1] * 384).tobytes()
        # Different content to ensure similarity difference if needed,
        # but here we use distinct IDs and lines for tie-breaking.
        self._insert_chunk("C1", "Process Data Alpha", line=1)
        self._insert_chunk("C2", "Process Data Beta", line=10)
        self._insert_chunk("C3", "Process Data Gamma", line=20)
        self._insert_chunk("H1", "def HandleEvents(x): return x", line=30)

    def _insert_chunk(self, chunk_id, content, line):
        doc_id = self.db.upsert_document("determinism.py", f"hash_{chunk_id}")
        self.db.insert_chunk({
            'id': chunk_id, 'document_id': doc_id, 'content': content,
            'content_hash': f"chash_{chunk_id}", 'start_line': line, 'end_line': line
        })
        self.db.insert_embedding(chunk_id, self.dummy_vector)

    def tearDown(self):
        self.db.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_repeatability(self):
        """Identical inputs produce identical outputs across repeated runs."""
        query = "Process Data"

        # Manually sync time for repeatability within the same process test
        fixed_time = "2024-01-01 12:00:00"
        for cid in ["C1", "C2", "C3", "H1"]:
            cursor = self.db.conn.cursor()
            cursor.execute("UPDATE retrieval_stats SET last_updated=? WHERE chunk_id=?", (fixed_time, cid))
            self.db.conn.commit()

        results_1 = retrieve(query, self.db, top_k=5)

        # Ensure log_event from the first run doesn't change success_score for the second run
        # in a way that breaks equality (since log_event updates stats).
        # We re-sync time to ensure recency is same.
        for cid in ["C1", "C2", "C3", "H1"]:
            cursor = self.db.conn.cursor()
            cursor.execute("UPDATE retrieval_stats SET last_updated=? WHERE chunk_id=?", (fixed_time, cid))
            self.db.conn.commit()

        results_2 = retrieve(query, self.db, top_k=5)

        # Deterministic normalization for comparison
        def normalize(results):
             return [(r["chunk_id"], r["score"]["final"]) for r in results]

        self.assertEqual(normalize(results_1), normalize(results_2))

    def test_stable_tie_breaker(self):
        """Ranking is stable even when scores are equal."""
        # Use a query that matches all "Process Data" equally
        query = "Process Data"

        # Ensure identical stats
        fixed_time = "2024-01-01 12:00:00"
        for cid in ["C1", "C2", "C3"]:
            cursor = self.db.conn.cursor()
            cursor.execute("UPDATE retrieval_stats SET success_score=1.0, last_updated=? WHERE chunk_id=?", (fixed_time, cid))
            self.db.conn.commit()

        results = retrieve(query, self.db, top_k=3)
        ids = [r['chunk_id'] for r in results]

        # Should be ['C1', 'C2', 'C3'] because tie-breaker is chunk_id ascending
        self.assertEqual(ids, ["C1", "C2", "C3"])

    def test_multi_hop_determinism(self):
        """Multi-hop expansion is stable and repeatable."""
        # Query that might trigger multi-hop if we reference HandleEvents
        query = "HandleEvents"

        results_1 = retrieve(query, self.db, top_k=2)
        results_2 = retrieve(query, self.db, top_k=2)

        ids_1 = [r['chunk_id'] for r in results_1]
        ids_2 = [r['chunk_id'] for r in results_2]

        self.assertEqual(ids_1, ids_2)

    def test_context_stitching_determinism(self):
        """Stitched context preserves line order regardless of retrieval order."""
        # Force C2 retrieval
        cursor = self.db.conn.cursor()
        cursor.execute("UPDATE retrieval_stats SET success_score=100.0 WHERE chunk_id='C2'")
        self.db.conn.commit()

        results = retrieve("Data", self.db, top_k=1)
        res = results[0]
        self.assertEqual(res['chunk_id'], "C2")

        context = res['context']
        self.assertIn("Alpha", context)
        self.assertIn("Beta", context)
        self.assertIn("Gamma", context)

        # Ordering check
        self.assertTrue(context.find("Alpha") < context.find("Beta"))
        self.assertTrue(context.find("Beta") < context.find("Gamma"))

if __name__ == "__main__":
    unittest.main()
