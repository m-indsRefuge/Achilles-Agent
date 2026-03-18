# src/memory_layer/short_term_memory.py

import os
import json
from typing import List, Dict, Optional
from .db.stm_db import STMStore


class ShortTermMemory:
    def __init__(
        self,
        max_size: int = 50,
        storage_path: str = "storage/short_term_memory.db",
    ):
        self.max_size = max_size
        self.storage_path = storage_path
        self.store = STMStore(storage_path)

    def add(self, entry: Dict):
        self.store.add(entry, max_size=self.max_size)

    def query(self, key: str, value: str, top_k: int = 5) -> List[Dict]:
        memory = self.store.get_all()
        matches = [
            item
            for item in memory
            if key in item and str(value).lower() in str(item[key] or "").lower()
        ]
        return matches[:top_k]

    def clear(self):
        self.store.clear()

    def summarize(self, summary_text: str):
        """
        Condense short-term memory by replacing old entries with a summary.
        """
        memory = self.store.get_all()
        if len(memory) < 2:
            return

        # Keep last 5 entries, replace others with summary
        keep_count = min(len(memory), 5)
        new_memory = [{"role": "system", "content": f"Summary of previous chat: {summary_text}"}] + memory[-keep_count:]

        self.store.clear()
        for entry in new_memory:
            self.store.add(entry, max_size=self.max_size)

    @property
    def memory(self):
        # Backward compatibility for existing logic
        return self.store.get_all()
