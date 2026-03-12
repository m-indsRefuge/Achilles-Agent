# src/memory_layer/short_term_memory.py

import os
import json
from typing import List, Dict, Optional


class ShortTermMemory:
    def __init__(
        self,
        max_size: int = 50,
        storage_path: str = "backend/memory_layer/storage/short_term_memory.json",
    ):
        self.max_size = max_size
        self.storage_path = storage_path
        self.memory: List[Dict] = []
        self._load()

    def add(self, entry: Dict):
        self.memory.append(entry)
        if len(self.memory) > self.max_size:
            self.memory.pop(0)
        self._persist()

    def query(self, key: str, value: str, top_k: int = 5) -> List[Dict]:
        matches = [
            item
            for item in self.memory
            if key in item and str(value).lower() in str(item[key] or "").lower()
        ]
        return matches[:top_k]

    def clear(self):
        self.memory = []
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)

    def _persist(self):
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error persisting STM: {e}")

    def _load(self):
        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                self.memory = json.load(f)
        except Exception as e:
            print(f"Error loading STM: {e}")
            self.memory = []
