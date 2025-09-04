# Simple memory manager for basic text storage and retrieval


class MemoryManager:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.entries = []
        self._load()

    def add_entry(self, text: str, metadata: dict | None = None):
        entry = {"text": text, "metadata": metadata or {}}
        self.entries.append(entry)
        self._persist()

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """
        Simple keyword search over stored entries.
        Returns top_k matching entries.
        """
        # Case-insensitive search
        matches = [
            entry
            for entry in self.entries
            if query_text.lower() in entry["text"].lower()
        ]
        return matches[:top_k]

    def _persist(self):
        import json, os

        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error persisting KnowledgeBase: {e}")

    def _load(self):
        import json, os

        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                self.entries = json.load(f)
        except Exception as e:
            print(f"Error loading KnowledgeBase: {e}")
            self.entries = []
