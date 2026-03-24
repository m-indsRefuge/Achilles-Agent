import unittest
import os
import time
import math
from core.storage import StorageManager
from core.retrieval import compute_dynamic_weights, normalize_query

class TestTimeDecayQueryFrequency(unittest.TestCase):
    def setUp(self):
        self.db_path = "backend/memory_layer/storage/decay_freq_test.sqlite"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.db = StorageManager(self.db_path)

    def tearDown(self):
        self.db.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_recency_bias(self):
        """Verify that recent queries produce higher frequency than old ones."""
        query = "Recency Test"
        norm_query = normalize_query(query)

        # 1. Insert OLD query events (10 days ago)
        old_time = time.time() - (10 * 86400)
        for _ in range(5):
            self.db.insert_retrieval_event(norm_query, [], [], timestamp=old_time)

        weights_old = compute_dynamic_weights(norm_query, self.db)
        print("Weights for 5 old queries:", weights_old)

        # 2. Insert RECENT query events (now)
        # Clear events first to isolate
        cursor = self.db.conn.cursor()
        cursor.execute("DELETE FROM retrieval_events")
        self.db.conn.commit()

        for _ in range(5):
            self.db.insert_retrieval_event(norm_query, [], [], timestamp=time.time())

        weights_recent = compute_dynamic_weights(norm_query, self.db)
        print("Weights for 5 recent queries:", weights_recent)

        self.assertGreater(weights_recent["feedback_weight"], weights_old["feedback_weight"])

    def test_adaptation_over_time(self):
        """Verify weight shifts toward similarity as queries age."""
        query = "Adaptation Test"
        norm_query = normalize_query(query)

        # Simulate repeated queries 10 days ago
        old_time = time.time() - (10 * 86400)
        for _ in range(10):
            self.db.insert_retrieval_event(norm_query, [], [], timestamp=old_time)

        weights = compute_dynamic_weights(norm_query, self.db)
        print("Initial weight (stale):", weights)

        # Because the queries are 10 days old, 10 * 86400 * 1e-5 = 8.64. exp(-8.64) is tiny.
        # frequency should be near 0
        self.assertEqual(weights["feedback_weight"], 0.1)

if __name__ == "__main__":
    unittest.main()
