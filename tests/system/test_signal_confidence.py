import unittest
import os
import time
from core.storage import StorageManager
from core.feedback import log_event, RetrievalEvent

class TestSignalConfidence(unittest.TestCase):
    def setUp(self):
        self.db_path = "backend/memory_layer/storage/signal_conf_test.sqlite"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.db = StorageManager(self.db_path)

        # Setup test chunks
        self.doc_id = self.db.upsert_document("test.py", "hash1")
        for cid in ["A", "B", "C", "dummy"]:
            self.db.insert_chunk({
                'id': cid, 'document_id': self.doc_id, 'content': f'content {cid}',
                'content_hash': f'hash_{cid}', 'start_line': 1, 'end_line': 1
            })

    def tearDown(self):
        self.db.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_strong_vs_weak_signals(self):
        """Verify that 'accept' signal increases score more than 'click' or 'view'."""
        print("\n--- TEST: Strong vs Weak Signals ---")

        # We need a shared retrieval set to see relative differences after normalization
        # 1. Event with A, B, C. Accept A, Click B, View C.
        from core.feedback import SIGNAL_WEIGHTS
        event = RetrievalEvent(
            "q", ["A", "B", "C"], ["A", "B", "C"],
            confidence_weights={
                "A": SIGNAL_WEIGHTS["accept"],
                "B": SIGNAL_WEIGHTS["click"],
                "C": SIGNAL_WEIGHTS["view"]
            }
        )
        log_event(event, self.db)

        all_stats = self.db.get_top_chunks(limit=10)
        stats = {s['chunk_id']: s['success_score'] for s in all_stats}
        print("Scores after calibrated signals:", stats)

        # Normalized scores should preserve the weight order: A > B > C
        self.assertGreater(stats["A"], stats["B"])
        self.assertGreater(stats["B"], stats["C"])

    def test_noise_resistance(self):
        """Verify that multiple low-confidence events have less impact than a single high-confidence event."""
        print("\n--- TEST: Noise Resistance ---")
        # Reset chunks
        cursor = self.db.conn.cursor()
        cursor.execute("UPDATE retrieval_stats SET success_score=0.1")
        self.db.conn.commit()

        # We must use separate sets to avoid cross-normalization in this comparison
        # 1. Single high-confidence 'accept' for A (in set {A, dummy})
        log_event(RetrievalEvent("q1", ["A", "dummy"], ["A"], signal_type="accept"), self.db)

        # Fetch current score of A
        score_a = next(s['success_score'] for s in self.db.get_top_chunks(limit=10) if s['chunk_id'] == 'A')

        # 2. Three low-confidence 'view' events for B (in set {B, dummy})
        for _ in range(3):
            log_event(RetrievalEvent("q2", ["B", "dummy"], ["B"], signal_type="view"), self.db)

        all_stats = self.db.get_top_chunks(limit=10)
        stats = {s['chunk_id']: s['success_score'] for s in all_stats}
        print("Final scores:", stats)

        # Accept (1.5) > 3x View (0.3 each).
        # Since they are in separate sets, both can be 1.0 (local max).
        # We check the RAW scores before normalization if possible, or use a common baseline.

        # Let's adjust test to use a shared set but different events
        cursor.execute("UPDATE retrieval_stats SET success_score=0.1")
        self.db.conn.commit()

        # Shared set {A, B, dummy}
        # 1. Accept A
        log_event(RetrievalEvent("q", ["A", "B", "dummy"], ["A"], signal_type="accept"), self.db)
        # 2. View B (multiple times)
        for _ in range(2):
            log_event(RetrievalEvent("q", ["A", "B", "dummy"], ["B"], signal_type="view"), self.db)

        stats = {s['chunk_id']: s['success_score'] for s in self.db.get_top_chunks(limit=10)}
        print("Final scores (shared set):", stats)
        self.assertGreater(stats["A"], stats["B"])

if __name__ == "__main__":
    unittest.main()
