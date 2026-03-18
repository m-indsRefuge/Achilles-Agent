import faiss
import numpy as np
import os
import json
from typing import List, Dict, Any, Optional

class QRStore:
    """
    FAISS-based storage for Quick Recall (working memory).
    Optimized for small, frequently updated context.
    """
    def __init__(self, storage_path: str, dimension: int = 384):
        self.storage_path = storage_path
        self.index_path = storage_path + ".index"
        self.meta_path = storage_path + ".meta"

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata: List[Dict[str, Any]] = []
        self.embeddings_cache: List[np.ndarray] = []

        self._load()

    def add(self, embedding: np.ndarray, entry: Dict[str, Any]):
        faiss.normalize_L2(embedding)
        self.index.add(embedding)
        self.metadata.append(entry)
        self.embeddings_cache.append(embedding.flatten())

        # Limit QR size to 100 recent entries
        if self.index.ntotal > 100:
            # Simple approach: clear and rebuild from last 100
            self.metadata = self.metadata[-100:]
            self.embeddings_cache = self.embeddings_cache[-100:]

            # Rebuild index
            self.index = faiss.IndexFlatIP(self.dimension)
            if self.embeddings_cache:
                embeddings_matrix = np.array(self.embeddings_cache).astype('float32')
                self.index.add(embeddings_matrix)

        self._save()

    def query(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                res = self.metadata[idx].copy()
                res["score"] = float(scores[0][i])
                results.append(res)
        return results

    def clear(self):
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        self.embeddings_cache = []
        if os.path.exists(self.index_path): os.remove(self.index_path)
        if os.path.exists(self.meta_path): os.remove(self.meta_path)
        if os.path.exists(self.storage_path): os.remove(self.storage_path)

    def _save(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def _load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            if self.index.ntotal > 0:
                try:
                    self.embeddings_cache = [self.index.reconstruct(i) for i in range(self.index.ntotal)]
                except:
                    pass
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
