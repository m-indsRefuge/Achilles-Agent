import unittest
import os
import time
from core.storage import StorageManager
from core.feedback import log_event, RetrievalEvent

class TestLocalCompetition(unittest.TestCase):
    def setUp(self):
        self.db_path = "backend/memory_layer/storage/competition_test.sqlite"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.db = StorageManager(self.db_path)

        # Setup dummy chunks A, B, C with equal score 10.0
        self.doc_id = self.db.upsert_document("test.py", "hash1")
        for cid in ["A", "B", "C"]:
            self.db.insert_chunk({
                'id': cid, 'document_id': self.doc_id, 'content': f'content {cid}',
                'content_hash': f'hash_{cid}', 'start_line': 1, 'end_line': 1
            })
            cursor = self.db.conn.cursor()
            cursor.execute("UPDATE retrieval_stats SET success_score=10.0 WHERE chunk_id=?", (cid,))
            self.db.conn.commit()

    def tearDown(self):
        self.db.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_relative_improvement(self):
        """Improving one result slightly reduces the relative standing of others."""
        print("\n--- TEST: Local Competition (Relative Improvement) ---")

        # Initial: A=10, B=10, C=10

        # Retrieve all 3, select ONLY A
        event = RetrievalEvent("query", ["A", "B", "C"], ["A"])
        log_event(event, self.db)

        stats = {s['chunk_id']: s['success_score'] for s in self.db.get_top_chunks(limit=3)}
        print("Scores after Selecting A:", stats)

        # A should increase (subject to momentum)
        self.assertGreater(stats["A"], 10.0)

        # B and C should decrease (subject to suppression)
        # previous was 10.0.
        # base_score after decay (0) and suppression (0.99) = 9.9
        # new_score = alpha * (9.9 + 0) + (1-alpha) * 10.0
        # = 0.3 * 9.9 + 0.7 * 10.0 = 2.97 + 7.0 = 9.97
        self.assertLess(stats["B"], 10.0)
        self.assertLess(stats["C"], 10.0)

        # Ranking should be A > B, A > C
        self.assertGreater(stats["A"], stats["B"])
        self.assertGreater(stats["A"], stats["C"])

    def test_repeated_competition(self):
        """Repeated selection should sharpen the ranking gap."""
        print("\n--- TEST: Repeated Competition ---")
        for _ in range(10):
            event = RetrievalEvent("query", ["A", "B", "C"], ["A"])
            log_event(event, self.db)

        stats = {s['chunk_id']: s['success_score'] for s in self.db.get_top_chunks(limit=3)}
        print("Final Scores:", stats)

        # Gap should be significant
        self.assertGreater(stats["A"] - stats["B"], 1.0)

if __name__ == "__main__":
    unittest.main()
