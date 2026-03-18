import os
import tempfile
import unittest
from typing import cast, Dict, Any
from memory_layer.short_term_memory import ShortTermMemory


class TestShortTermMemory(unittest.TestCase):
    def setUp(self):
        # Create a temporary file for persistence testing
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file_path = self.temp_file.name
        self.temp_file.close()
        self.memory = ShortTermMemory(max_size=3, storage_path=self.temp_file_path)

    def tearDown(self):
        if os.path.exists(self.temp_file_path):
            os.remove(self.temp_file_path)

    def test_add_and_size_limit(self):
        self.memory.add({"id": "1", "text": "first"})
        self.memory.add({"id": "2", "text": "second"})
        self.memory.add({"id": "3", "text": "third"})
        self.memory.add({"id": "4", "text": "fourth"})  # should evict first

        self.assertEqual(len(self.memory.memory), 3)
        self.assertEqual(len(self.memory.query("id", "1")), 0)  # first evicted
        results = self.memory.query("id", "4")
        self.assertTrue(len(results) > 0, "Newest item should exist")
        self.assertEqual(results[0]["text"], "fourth")

    def test_query_existing(self):
        self.memory.add({"id": "10", "text": "hello"})
        results = self.memory.query("id", "10")

        self.assertTrue(len(results) > 0, "Query should return an item")
        self.assertEqual(results[0]["text"], "hello")

    def test_persistence(self):
        self.memory.add({"id": "2", "text": "persist"})
        mem2 = ShortTermMemory(max_size=3, storage_path=self.temp_file_path)
        results = mem2.query("id", "2")

        self.assertTrue(len(results) > 0, "Query should return an item")
        self.assertEqual(results[0]["text"], "persist")

    def test_clear(self):
        self.memory.add({"id": 1, "text": "clear test"})
        self.memory.clear()
        self.assertEqual(len(self.memory.memory), 0)
        self.assertFalse(os.path.exists(self.temp_file_path))


if __name__ == "__main__":
    unittest.main()
