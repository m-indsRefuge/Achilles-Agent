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
            self.db.update_retrieval_stats(chunk_id, signal="selected")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT success_score FROM retrieval_stats WHERE chunk_id=?", (chunk_id,))
        score = cursor.fetchone()[0]
        conn.close()

        # Cap is 50.0. Due to alpha damping, it converges slowly.
        self.assertLessEqual(score, 50.0)
        self.assertGreater(score, 20.0)

    def test_slow_decay_behavior(self):
        # Verification of stability layer decay
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

        # Trigger update (retrieved but neutral)
        self.db.update_retrieval_stats(chunk_id, signal="neutral")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT success_score FROM retrieval_stats WHERE chunk_id=?", (chunk_id,))
        new_score = cursor.fetchone()[0]
        conn.close()

        # Score should decay slightly towards base but remain stable
        self.assertLess(new_score, 10.0)
        self.assertGreater(new_score, 5.0)

    def test_bias_resistance(self):
        # Verification that historical feedback influences rank
        # High lambda to demonstrate bias resistance in recency component
        fast_scorer = RetrievalScorer(decay_lambda=1.0)

        # High feedback, older
        now = time.time()
        old_high_score = {
            "success_score": 0.9,
            "last_updated": now - 20
        }

        # Low feedback, newer
        new_moderate_score = {
            "success_score": 0.1,
            "last_updated": now
        }

        # Equal similarity
        meta = {"raw_similarity": 0.5}

        score_old = fast_scorer.score(old_high_score, [], metadata=meta)
        score_new = fast_scorer.score(new_moderate_score, [], metadata=meta)

        # High feedback should win since it's 30% of score and recency is 0%
        self.assertGreater(score_old["final_score"], score_new["final_score"])

if __name__ == "__main__":
    unittest.main()
