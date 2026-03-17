# src/memory_layer/quick_recall.py

import os
import json
import numpy as np
import faiss
from typing import List, Dict, Optional
from .db.qr_index import QRStore


class QuickRecall:
    def __init__(self, storage_path: str, dimension: int = 384):
        if not storage_path:
            raise ValueError("storage_path must be provided")
        self.storage_path = storage_path
        self.store = QRStore(storage_path, dimension=dimension)

    @property
    def memory(self):
        return self.store.metadata

    def _persist(self):
        self.store._save()

    def add(
        self,
        entry: Dict,
        embedding: Optional[List[float]] = None,
        entry_type: str = "text",
    ):
        if embedding is None:
            return

        # Adjust store dimension if first entry has different dimension
        if len(embedding) != self.store.dimension:
            self.store = QRStore(self.storage_path, dimension=len(embedding))

        entry["type"] = entry_type
        emb_np = np.array([embedding], dtype="float32")
        self.store.add(emb_np, entry)

    def query(
        self, embedding: List[float], top_k: int = 5, entry_type: Optional[str] = None
    ) -> List[Dict]:
        if not embedding:
            return []

        query_emb = np.array([embedding], dtype="float32")
        return self.store.query(query_emb, top_k)

    def clear(self):
        self.store.clear()
