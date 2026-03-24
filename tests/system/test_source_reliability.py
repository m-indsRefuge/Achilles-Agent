import unittest
import os
import time
from core.storage import StorageManager
from core.feedback import log_event, RetrievalEvent

class TestSourceReliability(unittest.TestCase):
    def setUp(self):
        self.db_path = "backend/memory_layer/storage/source_reliability_test.sqlite"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.db = StorageManager(self.db_path)

        # Setup test chunks
        self.doc_id = self.db.upsert_document("test.py", "hash1")
        for cid in ["A", "B", "C", "D"]:
            self.db.insert_chunk({
                'id': cid, 'document_id': self.doc_id, 'content': f'content {cid}',
                'content_hash': f'hash_{cid}', 'start_line': 1, 'end_line': 1
            })

    def tearDown(self):
        self.db.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_reliable_vs_noisy_source(self):
        """Verify that a reliable source has more influence than a noisy one over time."""
        print("\n--- TEST: Reliable vs Noisy Source ---")

        # 1. 'Expert' user gives 20 good signals (accepts)
        for _ in range(20):
            log_event(RetrievalEvent("q", ["A"], ["A"], signal_type="accept", source_id="expert"), self.db)

        # 2. 'Noisy' user gives 20 bad signals (views without selection)
        for _ in range(20):
            log_event(RetrievalEvent("q", ["B"], [], signal_type="view", source_id="noisy"), self.db)

        expert_rel = self.db.get_source_reliability("expert")
        noisy_rel = self.db.get_source_reliability("noisy")

        print(f"Expert Reliability: {expert_rel}, Noisy Reliability: {noisy_rel}")
        self.assertGreater(expert_rel, 1.0)
        self.assertLess(noisy_rel, 1.0)

        # 3. Compare impact of a single click from both in a SHARED set
        # Reset scores to baseline 0.1 for comparison set
        cursor = self.db.conn.cursor()
        cursor.execute("UPDATE retrieval_stats SET success_score=0.1")
        self.db.conn.commit()

        # Event with C, D, and dummy. Click C (expert), Click D (noisy)
        # Since log_event applies one source per event, we do separate events
        # but in same retrieval set context
        log_event(RetrievalEvent("q", ["C", "D", "dummy"], ["C"], signal_type="click", source_id="expert"), self.db)
        log_event(RetrievalEvent("q", ["C", "D", "dummy"], ["D"], signal_type="click", source_id="noisy"), self.db)

        all_stats = self.db.get_top_chunks(limit=10)
        stats = {s['chunk_id']: s['success_score'] for s in all_stats}
        print(f"Impact: {stats}")

        # Expert effect (C) should be greater than noisy effect (D)
        self.assertGreater(stats["C"], stats["D"])

    def test_reliability_clamping(self):
        """Verify that reliability stays within [0.5, 1.5] bounds."""
        print("\n--- TEST: Reliability Clamping ---")

        # Expert 100 good signals
        for _ in range(100):
            log_event(RetrievalEvent("q", ["A"], ["A"], source_id="god_tier"), self.db)

        # Noisy 100 bad signals
        for _ in range(100):
            log_event(RetrievalEvent("q", ["B"], [], source_id="spammer"), self.db)

        self.assertEqual(self.db.get_source_reliability("god_tier"), 1.5)
        self.assertEqual(self.db.get_source_reliability("spammer"), 0.5)

if __name__ == "__main__":
    unittest.main()
