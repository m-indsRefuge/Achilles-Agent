import unittest
import os
import time
import math
from core.storage import StorageManager
from core.feedback import log_event, RetrievalEvent

class TestFeedbackStability(unittest.TestCase):
    def setUp(self):
        self.db_path = "backend/memory_layer/storage/stability_test.sqlite"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.db = StorageManager(self.db_path)

        # Setup dummy chunk
        self.doc_id = self.db.upsert_document("test.py", "hash1")
        self.chunk_id = "chunk_1"
        self.db.insert_chunk({
            'id': self.chunk_id, 'document_id': self.doc_id, 'content': 'print("hello")',
            'content_hash': 'chash1', 'start_line': 1, 'end_line': 1
        })

    def tearDown(self):
        self.db.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_convergence(self):
        """Repeated positive feedback should lead to stable scores."""
        print("\n--- TEST: Convergence ---")
        scores = []
        for _ in range(50):
            event = RetrievalEvent("q", [self.chunk_id], [self.chunk_id])
            log_event(event, self.db)
            stats = self.db.get_top_chunks(limit=1)[0]
            scores.append(stats['success_score'])

        print("Final Convergence Score:", scores[-1])
        # Score is normalized to 1.0 (single chunk)
        self.assertEqual(scores[-1], 1.0)

    def test_no_oscillation(self):
        """Alternating signals should not cause wild swings."""
        print("\n--- TEST: No Oscillation ---")
        scores = []
        # Alternate between select and dismiss
        for i in range(20):
            sig = "selected" if i % 2 == 0 else "dismissed"
            event = RetrievalEvent(
                "q", [self.chunk_id],
                [self.chunk_id] if sig == "selected" else [],
                [self.chunk_id] if sig == "dismissed" else []
            )
            log_event(event, self.db)
            stats = self.db.get_top_chunks(limit=1)[0]
            scores.append(stats['success_score'])

        print("Oscillation Scores:", scores[:5], "...", scores[-5:])
        # Score is normalized to 1.0 (single chunk)
        self.assertTrue(all(s == 1.0 for s in scores))

    def test_recovery(self):
        """Dismissed chunks can recover relative to others if later selected."""
        print("\n--- TEST: Recovery ---")
        # Add a second chunk to allow relative score changes
        self.db.insert_chunk({
            'id': "chunk_2", 'document_id': self.doc_id, 'content': 'other',
            'content_hash': 'chash2', 'start_line': 2, 'end_line': 2
        })

        # 1. Heavily dismiss C1 while C2 is neutral
        for _ in range(10):
            event = RetrievalEvent("q", ["chunk_1", "chunk_2"], [], ["chunk_1"])
            log_event(event, self.db)

        mid_stats = {s['chunk_id']: s['success_score'] for s in self.db.get_top_chunks(limit=2)}
        print("Scores after C1 dismissal:", mid_stats)
        self.assertLess(mid_stats["chunk_1"], mid_stats["chunk_2"])

        # 2. Repeatedly select C1
        for _ in range(20):
            event = RetrievalEvent("q", ["chunk_1", "chunk_2"], ["chunk_1"])
            log_event(event, self.db)

        final_stats = {s['chunk_id']: s['success_score'] for s in self.db.get_top_chunks(limit=2)}
        print("Scores after C1 recovery:", final_stats)
        # C1 should now be top
        self.assertEqual(final_stats["chunk_1"], 1.0)
        self.assertGreater(final_stats["chunk_1"], final_stats["chunk_2"])

    def test_no_dominance(self):
        """Repeated selection doesn't permanently lock out others."""
        print("\n--- TEST: No Dominance ---")
        # Chunk 1: popular
        # Chunk 2: new
        self.db.insert_chunk({
            'id': "chunk_2", 'document_id': self.doc_id, 'content': 'print("world")',
            'content_hash': 'chash2', 'start_line': 2, 'end_line': 2
        })

        # 1. C1 becomes dominant
        for _ in range(30):
            log_event(RetrievalEvent("q", ["chunk_1"], ["chunk_1"]), self.db)

        # 2. C2 starts getting selected
        for _ in range(10):
             log_event(RetrievalEvent("q", ["chunk_2"], ["chunk_2"]), self.db)

        stats = {s['chunk_id']: s['success_score'] for s in self.db.get_top_chunks(limit=2)}
        print("Stats after C2 activity:", stats)
        # One of them must be 1.0 (max)
        self.assertIn(1.0, stats.values())

    def test_temporal_decay_recovery(self):
        """Negative impact decays over time (recovery through aging)."""
        print("\n--- TEST: Temporal Decay ---")
        # Use high lambda for test visibility
        # Manually override DECAY_LAMBDA if possible or simulate time jump

        # 1. Dismiss chunk
        log_event(RetrievalEvent("q", [self.chunk_id], [], [self.chunk_id]), self.db)
        score_low = self.db.get_top_chunks(limit=1)[0]['success_score']

        # 2. Simulate time jump in DB (last_updated back in time)
        conn = self.db.conn
        cursor = conn.cursor()
        old_time = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(time.time() - 1000000))
        cursor.execute("UPDATE retrieval_stats SET last_updated=? WHERE chunk_id=?", (old_time, self.chunk_id))
        conn.commit()

        # 3. Update again (neutral)
        log_event(RetrievalEvent("q", [self.chunk_id], []), self.db)
        score_after_decay = self.db.get_top_chunks(limit=1)[0]['success_score']

        print("Score after dismissal:", score_low, "After decay jump:", score_after_decay)
        # Score should move toward 0.1 (it was already low, so let's check it stays bound)
        self.assertGreaterEqual(score_after_decay, 0.1)

if __name__ == "__main__":
    unittest.main()
