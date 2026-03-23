import unittest
import os
import time
from core.storage import StorageManager
from core.feedback import log_event, RetrievalEvent

class TestScoreNormalization(unittest.TestCase):
    def setUp(self):
        self.db_path = "backend/memory_layer/storage/normalization_test.sqlite"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.db = StorageManager(self.db_path)

        # Setup chunks A, B, C
        self.doc_id = self.db.upsert_document("test.py", "hash1")
        for cid, score in [("A", 10.0), ("B", 5.0), ("C", 2.0)]:
            self.db.insert_chunk({
                'id': cid, 'document_id': self.doc_id, 'content': f'content {cid}',
                'content_hash': f'hash_{cid}', 'start_line': 1, 'end_line': 1
            })
            cursor = self.db.conn.cursor()
            cursor.execute("UPDATE retrieval_stats SET success_score=? WHERE chunk_id=?", (score, cid))
            self.db.conn.commit()

    def tearDown(self):
        self.db.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_bounded_scores(self):
        """Verify that scores are scaled to [0.1, 1.0] after feedback."""
        print("\n--- TEST: Bounded Scores ---")

        # Retrieve and select A
        event = RetrievalEvent("q", ["A", "B", "C"], ["A"])
        log_event(event, self.db)

        stats = {s['chunk_id']: s['success_score'] for s in self.db.get_top_chunks(limit=3)}
        print("Scores after normalization:", stats)

        # Max score must be 1.0
        self.assertIn(1.0, stats.values())
        # All scores must be <= 1.0
        self.assertTrue(all(s <= 1.0 for s in stats.values()))
        # All scores must be >= 0.1
        self.assertTrue(all(s >= 0.1 for s in stats.values()))

    def test_ranking_preservation(self):
        """Verify that normalization doesn't change rank order."""
        print("\n--- TEST: Ranking Preservation ---")
        # Before: A(10), B(5), C(2) -> Order [A, B, C]

        # Retrieval (Neutral)
        event = RetrievalEvent("q", ["A", "B", "C"], [])
        log_event(event, self.db)

        stats = self.db.get_top_chunks(limit=3)
        order = [s['chunk_id'] for s in stats]
        print("Order after normalization:", order)

        self.assertEqual(order, ["A", "B", "C"])
        self.assertEqual(stats[0]['success_score'], 1.0) # A was max

    def test_floor_respect(self):
        """Verify that very small relative scores stay at 0.1."""
        print("\n--- TEST: Floor Respect ---")
        # Set A=100.0, C=0.001
        cursor = self.db.conn.cursor()
        cursor.execute("UPDATE retrieval_stats SET success_score=100.0 WHERE chunk_id='A'")
        cursor.execute("UPDATE retrieval_stats SET success_score=0.001 WHERE chunk_id='C'")
        self.db.conn.commit()

        event = RetrievalEvent("q", ["A", "C"], [])
        log_event(event, self.db)

        stats = {s['chunk_id']: s['success_score'] for s in self.db.get_top_chunks(limit=3)}
        print("Scores with extreme gap:", stats)

        # C should be 0.1, not 0.001/100 (0.00001)
        self.assertEqual(stats["C"], 0.1)

if __name__ == "__main__":
    unittest.main()
