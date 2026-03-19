from memory_layer.short_term_memory import ShortTermMemory
from memory_layer.quick_recall import QuickRecall
from memory_layer.knowledge_base import KnowledgeBase


class UnifiedMemory:
    def __init__(self, short_term_path, quick_recall_path, kb_path):
        self.kb = KnowledgeBase(storage_path=kb_path)
        self.short_term = ShortTermMemory(max_size=50, storage_path=short_term_path)
        self.quick_recall = QuickRecall(storage_path=quick_recall_path)

    # Add methods that wrap each memory layer
    def add_short_term(self, entry):
        self.short_term.add(entry)

    def summarize_short_term(self, summary_text):
        return self.short_term.summarize(summary_text)

    def add_quick_recall(self, entry, embedding=None):
        self.quick_recall.add(entry, embedding)

    def add_kb(self, text, metadata=None):
        return self.kb.add_entry(text, metadata)

    def add_kb_batch(self, texts, metadatas=None):
        return self.kb.add_entries(texts, metadatas)

    def clear_kb_file(self, file_path):
        return self.kb.clear_file_entries(file_path)

    def clear_all(self):
        self.short_term.clear()
        self.quick_recall.clear()
        self.kb.clear()

    def query_short_term(self, key, value):
        return self.short_term.query(key, value)

    def query_quick_recall(self, embedding, top_k=5):
        return self.quick_recall.query(embedding, top_k)

    def query_kb(self, query_text, top_k=5):
        return self.kb.search(query_text, top_k)

    def rerank_kb(self, query_text, results):
        return self.kb.rerank(query_text, results)
