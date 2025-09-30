# src/retrieval/indexer.py
from typing import List, Tuple, Dict, Optional
import numpy as np
import hnswlib
import os
import json


class HnswIndexer:
    """
    Lightweight HNSW indexer wrapper using hnswlib.
    Stores metadata separately in a JSONL file mapping id -> metadata.
    """

    def __init__(
        self,
        dim: int,
        index_path: str = "data/faiss_index",
        max_elements: int = 100_000,
        ef: int = 200,
        M: int = 16,
    ):
        self.dim = dim
        self.index_path = index_path
        self.max_elements = max_elements
        self._pfx = index_path
        os.makedirs(os.path.dirname(self._pfx), exist_ok=True)
        self.ef = ef
        self.M = M
        self._index = hnswlib.Index(space="cosine", dim=dim)
        self._initialized = False
        self._id_to_meta: Dict[int, Dict] = {}
        self._meta_path = f"{self._pfx}.meta.json"

    def init_index(self):
        if not self._initialized:
            self._index.init_index(
                max_elements=self.max_elements, ef_construction=self.ef, M=self.M
            )
            self._index.set_ef(self.ef)
            self._initialized = True

    def add(
        self, ids: List[int], vectors: np.ndarray, metas: Optional[List[Dict]] = None
    ):
        if not self._initialized:
            self.init_index()
        vectors = np.asarray(vectors, dtype=np.float32)
        self._index.add_items(vectors, ids)
        if metas:
            for _id, meta in zip(ids, metas):
                self._id_to_meta[int(_id)] = meta
        self._save_meta()

    def search(self, vector: np.ndarray, k: int = 10) -> Tuple[List[int], List[float]]:
        """
        vector: shape (dim,) or (1,dim)
        returns: (ids, distances)
        """
        v = np.asarray(vector, dtype=np.float32)
        if v.ndim == 1:
            v = v.reshape(1, -1)
        labels, distances = self._index.knn_query(v, k=k)
        # labels: shape (1, k)
        return labels[0].tolist(), distances[0].tolist()

    def get_meta(self, ids: List[int]) -> List[Dict]:
        return [self._id_to_meta.get(int(i), {}) for i in ids]

    def save(self, path_prefix: Optional[str] = None):
        pfx = path_prefix or self._pfx
        os.makedirs(os.path.dirname(pfx), exist_ok=True)
        self._index.save_index(f"{pfx}.hnsw")
        with open(f"{pfx}.meta.json", "w", encoding="utf-8") as fh:
            json.dump(self._id_to_meta, fh)

    def load(self, path_prefix: Optional[str] = None):
        pfx = path_prefix or self._pfx
        idx_file = f"{pfx}.hnsw"
        meta_file = f"{pfx}.meta.json"
        if not os.path.exists(idx_file):
            raise FileNotFoundError(idx_file)
        self._index.load_index(idx_file, max_elements=self.max_elements)
        self._initialized = True
        if os.path.exists(meta_file):
            with open(meta_file, "r", encoding="utf-8") as fh:
                self._id_to_meta = json.load(fh)
        else:
            self._id_to_meta = {}

    def _save_meta(self):
        with open(self._meta_path, "w", encoding="utf-8") as fh:
            json.dump(self._id_to_meta, fh)
