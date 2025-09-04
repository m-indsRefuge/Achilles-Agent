from memory_layer.short_term_memory import ShortTermMemory
from memory_layer.quick_recall import QuickRecall
from memory_layer.MemoryManager import MemoryManager


class UnifiedMemory:
    def __init__(self, short_term_path, quick_recall_path, kb_path):
        self.short_term = ShortTermMemory(max_size=50, storage_path=short_term_path)
        self.quick_recall = QuickRecall(storage_path=quick_recall_path)
        self.kb = MemoryManager(storage_path=kb_path)

    # Add methods that wrap each memory layer
    def add_short_term(self, entry):
        self.short_term.add(entry)

    def add_quick_recall(self, entry, embedding=None):
        self.quick_recall.add(entry, embedding)

    def add_kb(self, text, metadata=None):
        self.kb.add_entry(text, metadata)

    def query_short_term(self, key, value):
        return self.short_term.query(key, value)

    def query_quick_recall(self, embedding, top_k=5):
        return self.quick_recall.query(embedding, top_k)

    def query_kb(self, query_text, top_k=5):
        return self.kb.query(query_text, top_k)
