import unittest
import os
import array
from core.storage import StorageManager
from core.retrieval import retrieve, compute_dynamic_weights, normalize_query
from core.feedback import log_event, RetrievalEvent

class TestDynamicWeighting(unittest.TestCase):
    def setUp(self):
        self.db_path = "backend/memory_layer/storage/dynamic_test.sqlite"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.db = StorageManager(self.db_path)

        # Setup test data
        self.dummy_vector = array.array('f', [0.1] * 384).tobytes()
        self._insert_chunk("HighSim", "Target content Alpha", line=1, score=0.1)
        self._insert_chunk("HighFeed", "Other content Beta", line=10, score=1.0)

    def _insert_chunk(self, chunk_id, content, line, score=1.0):
        doc_id = self.db.upsert_document("test.py", f"hash_{chunk_id}")
        self.db.insert_chunk({
            'id': chunk_id, 'document_id': doc_id, 'content': content,
            'content_hash': f"chash_{chunk_id}", 'start_line': line, 'end_line': line
        })
        self.db.insert_embedding(chunk_id, self.dummy_vector)
        cursor = self.db.conn.cursor()
        cursor.execute("UPDATE retrieval_stats SET success_score=? WHERE chunk_id=?", (score, chunk_id))
        self.db.conn.commit()

    def tearDown(self):
        self.db.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_weight_progression(self):
        """Verify that feedback weight increases for repeated queries."""
        query = "Target content"
        norm_query = normalize_query(query)

        # Initial: 0 repetition
        weights = compute_dynamic_weights(norm_query, self.db)
        print("Initial weights:", weights)
        self.assertEqual(weights["feedback_weight"], 0.1)

        # Repeat 2 times
        for _ in range(2):
            event = RetrievalEvent(norm_query, ["HighSim"], ["HighSim"])
            log_event(event, self.db)

        weights = compute_dynamic_weights(norm_query, self.db)
        print("Weights after 2 repeats:", weights)
        # 0.1 + 0.1 * 2 = 0.3
        self.assertEqual(weights["feedback_weight"], 0.3)

        # Repeat until cap
        for _ in range(10):
            event = RetrievalEvent(norm_query, ["HighSim"], ["HighSim"])
            log_event(event, self.db)

        weights = compute_dynamic_weights(norm_query, self.db)
        print("Weights after many repeats (capped):", weights)
        self.assertEqual(weights["feedback_weight"], 0.5)

    def test_ranking_shift(self):
        """Verify that higher feedback ranks higher as its weight increases."""
        # Query that matches both (in this setup both have same dummy vector,
        # so similarity is equal)
        query = "content"
        norm_query = normalize_query(query)

        # Initially feedback weight is 0.1.
        # Since similarities are equal, higher feedback (HighFeed) should always win,
        # but let's see how scores change.

        res1 = retrieve(query, self.db, top_k=2)
        score1_highfeed = next(r['score']['final'] for r in res1 if r['chunk_id'] == "HighFeed")

        # Repeat query 4 times to boost feedback weight to 0.5
        for _ in range(4):
            log_event(RetrievalEvent(norm_query, ["HighFeed"], ["HighFeed"]), self.db)

        res2 = retrieve(query, self.db, top_k=2)
        score2_highfeed = next(r['score']['final'] for r in res2 if r['chunk_id'] == "HighFeed")

        print(f"HighFeed score: initial={score1_highfeed}, boosted_weight={score2_highfeed}")
        # HighFeed (success_score=1.0)
        # initial: 0.9 * Sim + 0.1 * 1.0
        # boosted: 0.5 * Sim + 0.5 * 1.0
        # Since Sim < 1.0 (it's 0.1 normalized to (0.1+1)/2 = 0.55),
        # increasing weight of feedback(1.0) vs similarity(0.55) increases total score.
        self.assertGreater(score2_highfeed, score1_highfeed)

if __name__ == "__main__":
    unittest.main()
