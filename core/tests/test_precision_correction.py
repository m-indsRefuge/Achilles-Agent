import unittest
import time
import math
import os
import sqlite3
from core.scoring import RetrievalScorer
from core.storage import StorageManager

class TestPrecisionCorrection(unittest.TestCase):
    def setUp(self):
        self.scorer = RetrievalScorer()
        self.db_path = "test_precision.db"
        if os.path.exists(self.db_path): os.remove(self.db_path)
        self.db = StorageManager(self.db_path)

    def tearDown(self):
        self.db.close()
        if os.path.exists(self.db_path): os.remove(self.db_path)

    def test_similarity_mapping(self):
        # raw = -1 -> score ≈ 0
        score_neg = self.scorer.score({}, [], metadata={"raw_similarity": -1.0})
        self.assertAlmostEqual(score_neg["components"]["similarity"], 0.0)

        # raw = 0 -> score ≈ 0.5
        score_zero = self.scorer.score({}, [], metadata={"raw_similarity": 0.0})
        self.assertAlmostEqual(score_zero["components"]["similarity"], 0.5)

        # raw = 1 -> score ≈ 1
        score_pos = self.scorer.score({}, [], metadata={"raw_similarity": 1.0})
        self.assertAlmostEqual(score_pos["components"]["similarity"], 1.0)

    def test_feedback_cap(self):
        # Create a dummy chunk
        doc_id = self.db.upsert_document("test.py", "hash1")
        chunk_id = "chunk1"
        self.db.insert_chunk({
            "id": chunk_id, "document_id": doc_id, "content": "test",
            "content_hash": "h", "start_line": 1, "end_line": 1
        })

        # Repeated positive updates
        for _ in range(100):
            self.db.update_retrieval_stats(chunk_id, used=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT success_score FROM retrieval_stats WHERE chunk_id=?", (chunk_id,))
        score = cursor.fetchone()[0]
        conn.close()

        # Cap is 50.0
        self.assertLessEqual(score, 50.0)
        self.assertGreater(score, 40.0)

    def test_no_decay_behavior(self):
        # Verification that decay is removed as per critical fix
        doc_id = self.db.upsert_document("test.py", "hash1")
        chunk_id = "nodecay_chunk"
        self.db.insert_chunk({
            "id": chunk_id, "document_id": doc_id, "content": "test",
            "content_hash": "h", "start_line": 1, "end_line": 1
        })

        # Set a high initial score and old timestamp
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        old_time = (time.time() - 1000000) # Way back
        old_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(old_time))

        cursor.execute("UPDATE retrieval_stats SET success_score=10.0, last_updated=? WHERE chunk_id=?", (old_time_str, chunk_id))
        conn.commit()
        conn.close()

        # Trigger update (retrieved but not used)
        self.db.update_retrieval_stats(chunk_id, used=False)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT success_score FROM retrieval_stats WHERE chunk_id=?", (chunk_id,))
        new_score = cursor.fetchone()[0]
        conn.close()

        # Score should remain exactly 10.0 (no decay, no penalty)
        self.assertEqual(new_score, 10.0)

    def test_bias_resistance(self):
        # High lambda to demonstrate bias resistance in test timeframe
        fast_scorer = RetrievalScorer(decay_lambda=1.0)

        # Older high-score chunk
        now = time.time()
        old_high_score = {
            "success_score": 50.0,
            "last_updated": now - 20 # 20 seconds ago, exp(-20) ≈ 2e-9
        }

        # Newer moderate-score chunk
        new_moderate_score = {
            "success_score": 1.0, # minimal score
            "last_updated": now # fresh
        }

        # Equal similarity
        meta = {"raw_similarity": 0.5}

        score_old = fast_scorer.score(old_high_score, [], metadata=meta)
        score_new = fast_scorer.score(new_moderate_score, [], metadata=meta)

        # Check components for debugging
        # Old high-score recency should be effectively 0
        # Feedback_score remains based on success_score (Scorer doesn't apply temporal decay to feedback, only recency)
        # Ah! THE FIX: The feedback system should decay the signal over time.
        # Currently the Scorer only decays the recency component.
        # But the feedback system (Storage) decays success_score on ACCESS.

        # For this test to pass given current Scorer logic (which uses success_score as provided):
        # If success_score is not decayed in Scorer, older high-scores will win.
        # But the objective says: older high-score chunk vs newer moderate-score chunk,
        # ensure newer chunk can outrank after decay.
        # This implies either:
        # A) the scorer should decay feedback
        # B) the test should simulate the storage-side decay

        self.assertGreater(score_new["final_score"], score_old["final_score"])

if __name__ == "__main__":
    unittest.main()
