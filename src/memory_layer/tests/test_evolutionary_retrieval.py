import unittest
import os
import sys
import array
# Add core to path
current_dir = os.path.dirname(os.path.abspath(__file__))
core_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "core")
sys.path.append(core_path)

from memory_layer.knowledge_base import KnowledgeBase

class TestEvolutionaryRetrieval(unittest.TestCase):
    def setUp(self):
        self.db_path = "backend/memory_layer/storage/test_evolutionary.sqlite"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.kb = KnowledgeBase(self.db_path)

    def test_multi_hop_retrieval(self):
        # Create a dependency chain
        # Chunk A mentions 'CalculateFactorial'
        # Chunk B defines 'CalculateFactorial' and mentions 'MathUtils'
        # Chunk C defines 'MathUtils'

        self.kb.add_entries([
            "I need to use CalculateFactorial for the computation.",
            "def CalculateFactorial(n): return MathUtils.prod(range(1, n+1))",
            "class MathUtils: @staticmethod def prod(items): return 1"
        ], [
            {"path": "usage.py"},
            {"path": "logic.py"},
            {"path": "utils.py"}
        ])

        # Search for 'computation'
        # Without multi-hop, we might only get chunk 1.
        # With multi-hop, it extracts 'CalculateFactorial' and finds chunk 2.
        results = self.kb.search("computation", expand_context=True)

        chunk_texts = [r["text"] for r in results]
        self.assertTrue(any("CalculateFactorial" in t for t in chunk_texts), "Should find the definition via multi-hop")

    def test_context_expansion(self):
        # Add chunks in sequence
        self.kb.add_entries([
            "First part of a long function.",
            "Second part of the same function.",
            "Third part of the same function."
        ], [
            {"path": "long_file.py", "lineStart": 1},
            {"path": "long_file.py", "lineStart": 10},
            {"path": "long_file.py", "lineStart": 20}
        ])

        # Search for 'Second part'
        results = self.kb.search("Second part", expand_context=True)

        # The 'context' field should contain the neighbors
        self.assertTrue(any("First part" in r.get("text", "") or "First part" in r.get("context", "") for r in results))
        self.assertTrue(any("Third part" in r.get("text", "") or "Third part" in r.get("context", "") for r in results))

if __name__ == "__main__":
    unittest.main()
