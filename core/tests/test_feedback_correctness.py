import unittest
import os
import time
from core.storage import StorageManager
from core.feedback import log_event, RetrievalEvent

class TestFeedbackCorrectness(unittest.TestCase):
    def setUp(self):
        self.db_path = "backend/memory_layer/storage/test_feedback.sqlite"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.db = StorageManager(self.db_path)

        # Setup dummy chunk
        self.doc_id = self.db.upsert_document("test.py", "hash1")
        self.chunk_id = "chunk_1"
        self.db.insert_chunk({
            'id': self.chunk_id,
            'document_id': self.doc_id,
            'content': 'print("hello")',
            'content_hash': 'chash1',
            'start_line': 1,
            'end_line': 1
        })

    def test_positive_reinforcement(self):
        # Initial score is 1.0
        stats = self.db.get_top_chunks(limit=1)[0]
        self.assertEqual(stats['success_score'], 1.0)

        # Simulate selection
        event = RetrievalEvent("query", [self.chunk_id], [self.chunk_id], [])
        log_event(event, self.db)

        # Score should increase
        stats = self.db.get_top_chunks(limit=1)[0]
        self.assertEqual(stats['success_score'], 2.0)
        self.assertEqual(stats['retrieval_count'], 1)

    def test_no_penalty_on_retrieval(self):
        # Initial score is 1.0

        # Simulate retrieval without selection
        event = RetrievalEvent("query", [self.chunk_id], [], [])
        log_event(event, self.db)

        # Score should remain 1.0 (no penalty)
        stats = self.db.get_top_chunks(limit=1)[0]
        self.assertEqual(stats['success_score'], 1.0)
        self.assertEqual(stats['retrieval_count'], 1)

    def test_negative_reinforcement(self):
        # Simulate dismissal
        event = RetrievalEvent("query", [self.chunk_id], [], [self.chunk_id])
        log_event(event, self.db)

        # Score should decrease
        stats = self.db.get_top_chunks(limit=1)[0]
        self.assertEqual(stats['success_score'], 0.8)

    def test_stability_over_time(self):
        # Repeated retrieval without selection should not degrade score
        for _ in range(5):
            event = RetrievalEvent("query", [self.chunk_id], [], [])
            log_event(event, self.db)

        stats = self.db.get_top_chunks(limit=1)[0]
        self.assertEqual(stats['success_score'], 1.0)
        self.assertEqual(stats['retrieval_count'], 5)

    def tearDown(self):
        self.db.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

if __name__ == "__main__":
    unittest.main()
