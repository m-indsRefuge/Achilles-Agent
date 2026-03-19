import unittest
import time
import math
from core.scoring import RetrievalScorer

class TestScoring(unittest.TestCase):
    def setUp(self):
        self.scorer = RetrievalScorer()

    def test_similarity_ranking(self):
        # Higher similarity -> higher rank
        chunk = {"success_score": 1.0, "created_at": time.time()}

        score_high = self.scorer.score(chunk, [], metadata={"raw_similarity": 0.9})
        score_low = self.scorer.score(chunk, [], metadata={"raw_similarity": 0.3})

        self.assertGreater(score_high["final_score"], score_low["final_score"])
        self.assertEqual(score_high["components"]["similarity"], 0.9)

    def test_feedback_impact(self):
        # High feedback improves rank
        chunk_high = {"success_score": 10.0, "created_at": time.time()}
        chunk_low = {"success_score": 1.0, "created_at": time.time()}

        # Keep similarity same
        meta = {"raw_similarity": 0.5}

        score_high = self.scorer.score(chunk_high, [], metadata=meta)
        score_low = self.scorer.score(chunk_low, [], metadata=meta)

        self.assertGreater(score_high["final_score"], score_low["final_score"])
        # Check bounded nature
        self.assertLess(score_high["components"]["feedback"], 1.0)

    def test_recency_decay(self):
        # Old chunks decay
        now = time.time()
        chunk_new = {"success_score": 1.0, "created_at": now}
        chunk_old = {"success_score": 1.0, "created_at": now - 100000}

        meta = {"raw_similarity": 0.5}

        score_new = self.scorer.score(chunk_new, [], metadata=meta)
        score_old = self.scorer.score(chunk_old, [], metadata=meta)

        self.assertGreater(score_new["final_score"], score_old["final_score"])
        self.assertLess(score_old["components"]["recency"], 1.0)

    def test_output_structure(self):
        chunk = {"success_score": 1.0, "created_at": time.time()}
        score = self.scorer.score(chunk, [], metadata={"raw_similarity": 0.5})

        self.assertIn("final_score", score)
        self.assertIn("components", score)
        self.assertIn("similarity", score["components"])
        self.assertIn("feedback", score["components"])
        self.assertIn("recency", score["components"])

if __name__ == "__main__":
    unittest.main()
