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

        # Scorer uses (raw + 1.0) / 2.0 normalization
        score_high = self.scorer.score(chunk, [], metadata={"raw_similarity": 0.9})
        score_low = self.scorer.score(chunk, [], metadata={"raw_similarity": 0.3})

        self.assertGreater(score_high["final_score"], score_low["final_score"])
        # Expected similarity component: (0.9 + 1.0) / 2.0 = 0.95
        self.assertEqual(score_high["components"]["similarity"], 0.95)

    def test_feedback_impact(self):
        # High feedback improves rank
        chunk_high = {"success_score": 0.9, "created_at": time.time()}
        chunk_low = {"success_score": 0.1, "created_at": time.time()}

        # Keep similarity same
        meta = {"raw_similarity": 0.5}

        score_high = self.scorer.score(chunk_high, [], metadata=meta)
        score_low = self.scorer.score(chunk_low, [], metadata=meta)

        self.assertGreater(score_high["final_score"], score_low["final_score"])
        # Check bounded nature
        self.assertLessEqual(score_high["components"]["feedback"], 1.0)

    def test_recency_component(self):
        # Old chunks decay in recency component
        now = time.time()
        chunk_new = {"success_score": 1.0, "created_at": now}
        chunk_old = {"success_score": 1.0, "created_at": now - 100000}

        meta = {"raw_similarity": 0.5}

        score_new = self.scorer.score(chunk_new, [], metadata=meta)
        score_old = self.scorer.score(chunk_old, [], metadata=meta)

        # Note: recency no longer contributes to final_score but is in components
        self.assertLess(score_old["components"]["recency"], 1.0)
        self.assertGreater(score_new["components"]["recency"], score_old["components"]["recency"])

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
