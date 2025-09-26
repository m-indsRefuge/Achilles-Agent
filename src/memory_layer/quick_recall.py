# src/memory_layer/quick_recall.py

import os
import json
import numpy as np
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity


class QuickRecall:
    def __init__(self, storage_path: str):
        if not storage_path:
            raise ValueError("storage_path must be provided")
        self.storage_path = storage_path
        self.memory: List[Dict] = []
        self._load()

    def add(
        self,
        entry: Dict,
        embedding: Optional[List[float]] = None,
        entry_type: str = "text",
    ):
        entry["type"] = entry_type
        if embedding is not None:
            entry["embedding"] = np.array(embedding, dtype=float)
        self.memory.append(entry)
        self._persist()

    def query(
        self, embedding: List[float], top_k: int = 5, entry_type: Optional[str] = None
    ) -> List[Dict]:
        mem = self.memory
        if entry_type:
            mem = [e for e in self.memory if e.get("type") == entry_type]
        if not mem:
            return []

        memory_embeddings = np.array([item["embedding"] for item in mem])
        query_embedding = np.array(embedding).reshape(1, -1)

        similarities = cosine_similarity(query_embedding, memory_embeddings)[0]
        top_indices = similarities.argsort()[::-1][:top_k]
        return [mem[i] for i in top_indices]

    def clear(self):
        self.memory = []
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)

    def _persist(self):
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(
                    [self._serialize_entry(e) for e in self.memory],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as e:
            print(f"Error persisting QuickRecall memory: {e}")

    def _load(self):
        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                self.memory = [self._deserialize_entry(e) for e in loaded]
        except Exception as e:
            print(f"Error loading QuickRecall memory: {e}")
            self.memory = []

    @staticmethod
    def _serialize_entry(entry: Dict) -> Dict:
        e_copy = entry.copy()
        if "embedding" in e_copy and isinstance(e_copy["embedding"], np.ndarray):
            e_copy["embedding"] = e_copy["embedding"].tolist()
        return e_copy

    @staticmethod
    def _deserialize_entry(entry: Dict) -> Dict:
        e_copy = entry.copy()
        if "embedding" in e_copy and isinstance(e_copy["embedding"], list):
            e_copy["embedding"] = np.array(e_copy["embedding"], dtype=float)
        return e_copy
