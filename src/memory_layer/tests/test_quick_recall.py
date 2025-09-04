import os
import tempfile
import unittest
from memory_layer.quick_recall import QuickRecall


class TestQuickRecall(unittest.TestCase):
    def setUp(self):
        # Create a temporary file for persistence
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.storage_path = self.temp_file.name
        self.temp_file.close()
        self.memory = QuickRecall(storage_path=self.storage_path)

    def tearDown(self):
        # Cleanup temp file if it exists
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)

    def test_add_and_retrieve(self):
        # Add entries with embeddings
        factorial_embedding = [1.0, 0.0, 0.0]
        sort_embedding = [0.0, 1.0, 0.0]
        self.memory.add({"text": "Python function to compute factorial."}, factorial_embedding)
        self.memory.add({"text": "Sort a list in ascending order."}, sort_embedding)

        # Quick retrieval test
        results = self.memory.query(factorial_embedding)
        self.assertTrue(len(results) > 0)
        self.assertIn("factorial", results[0]["text"])

        results = self.memory.query(sort_embedding)
        self.assertTrue(len(results) > 0)
        self.assertIn("Sort", results[0]["text"])

    def test_persistence(self):
        # Add entries and persist
        persisted_embedding = [0.5, 0.5, 0.0]
        self.memory.add({"text": "Persisted entry test."}, persisted_embedding)
        self.memory._persist()

        # Load memory from disk
        new_memory = QuickRecall(storage_path=self.storage_path)
        results = new_memory.query(persisted_embedding)
        self.assertTrue(len(results) > 0)
        self.assertIn("Persisted entry test.", results[0]["text"])

    def test_clear(self):
        temp_embedding = [0.0, 0.0, 1.0]
        self.memory.add({"text": "Temporary entry"}, temp_embedding)
        self.memory.clear()
        self.assertEqual(len(self.memory.memory), 0)
        self.assertFalse(os.path.exists(self.storage_path))


if __name__ == "__main__":
    unittest.main()
