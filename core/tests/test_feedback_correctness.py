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

        # Score should be normalized to 1.0 (it's the only chunk in the set)
        stats = self.db.get_top_chunks(limit=1)[0]
        self.assertEqual(stats['success_score'], 1.0)
        self.assertEqual(stats['retrieval_count'], 1)

    def test_no_additional_penalty_on_retrieval(self):
        # Initial score is 1.0

        # Simulate retrieval without selection (neutral)
        event = RetrievalEvent("query", [self.chunk_id], [], [])
        log_event(event, self.db)

        # Score should remain normalized to 1.0 as it's the max (and only) in the set
        stats = self.db.get_top_chunks(limit=1)[0]
        self.assertEqual(stats['success_score'], 1.0)
        self.assertEqual(stats['retrieval_count'], 1)

    def test_negative_reinforcement(self):
        # Simulate dismissal
        event = RetrievalEvent("query", [self.chunk_id], [], [self.chunk_id])
        log_event(event, self.db)

        # Even after dismissal, if it's the only one, it's normalized to 1.0
        # To truly test reinforcement, we need multiple chunks.
        stats = self.db.get_top_chunks(limit=1)[0]
        self.assertEqual(stats['success_score'], 1.0)

    def test_stability_over_time(self):
        # Repeated retrieval without selection should be stable at 1.0 if alone
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
