import unittest
from memory_layer.knowledge_base import KnowledgeBase


class TestKnowledgeBase(unittest.TestCase):

    def setUp(self):
        self.kb = KnowledgeBase("backend/memory_layer/storage/test_kb.json")

    def test_add_and_search(self):
        entry_id = self.kb.add_entry("A Python function that calculates factorial.")
        results = self.kb.search("factorial function")
        self.assertTrue(any("factorial" in r["text"] for r in results))

    def test_update_entry(self):
        entry_id = self.kb.add_entry("Old text")
        updated = self.kb.update_entry(entry_id, "New text about recursion")
        self.assertTrue(updated)
        results = self.kb.search("recursion")
        self.assertTrue(any("recursion" in r["text"] for r in results))


if __name__ == "__main__":
    unittest.main()
