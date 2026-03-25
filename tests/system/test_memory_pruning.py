import unittest
import os
import time
import math
from core.storage import StorageManager
from core.retrieval import retrieve

class TestMemoryPruning(unittest.TestCase):
    def setUp(self):
        self.db_path = "backend/memory_layer/storage/pruning_test.sqlite"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.db = StorageManager(self.db_path)

        # Setup test data: 15 chunks (enough to exceed floor of 10)
        self.doc_id = self.db.upsert_document("test.py", "hash1")
        for i in range(15):
            cid = f"C{i}"
            self.db.insert_chunk({
                'id': cid, 'document_id': self.doc_id, 'content': f'content {i}',
                'content_hash': f'hash_{i}', 'start_line': i, 'end_line': i
            })

    def tearDown(self):
        self.db.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_low_value_removal(self):
        """Verify that low-score and old chunks are pruned."""
        print("\n--- TEST: Low Value Removal ---")

        # 1. Mark C0 as OLD and LOW SCORE
        # success_score=0.1
        # timestamp 100 days ago
        old_time = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(time.time() - (100 * 86400)))
        cursor = self.db.conn.cursor()
        cursor.execute("UPDATE retrieval_stats SET success_score=0.1, last_updated=? WHERE chunk_id='C0'", (old_time,))
        self.db.conn.commit()

        # 2. Trigger Pruning
        self.db.prune_memory()

        # 3. Verify C0 is inactive
        cursor.execute("SELECT is_active FROM chunks WHERE id='C0'")
        self.assertEqual(cursor.fetchone()[0], 0)

        # Verify active count decreased (at least one pruned)
        active_chunks = self.db.fetch_active_chunks()
        self.assertLess(len(active_chunks), 15)

    def test_high_value_retention(self):
        """Verify that high-success chunks are protected even if old."""
        print("\n--- TEST: High Value Retention ---")

        # 1. Mark C1 as OLD but HIGH SCORE
        old_time = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(time.time() - (100 * 86400)))
        cursor = self.db.conn.cursor()
        cursor.execute("UPDATE retrieval_stats SET success_score=1.0, last_updated=? WHERE chunk_id='C1'", (old_time,))
        self.db.conn.commit()

        # 2. Trigger Pruning
        self.db.prune_memory()

        # 3. Verify C1 remains active (protected by success_score > 0.8)
        cursor.execute("SELECT is_active FROM chunks WHERE id='C1'")
        self.assertEqual(cursor.fetchone()[0], 1)

    def test_floor_protection(self):
        """Verify that system never prunes below the minimum floor (10)."""
        print("\n--- TEST: Floor Protection ---")

        # 1. Make all chunks low value
        old_time = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(time.time() - (100 * 86400)))
        cursor = self.db.conn.cursor()
        # Explicitly set low scores for all chunks to ensure retention_score < 0.05
        cursor.execute("UPDATE retrieval_stats SET success_score=0.0001, last_updated=?", (old_time,))
        self.db.conn.commit()

        # 2. Trigger Pruning
        self.db.prune_memory()

        # 3. Verify exactly 10 remain (floor)
        active_chunks = self.db.fetch_active_chunks()
        self.assertEqual(len(active_chunks), 10)

if __name__ == "__main__":
    unittest.main()
