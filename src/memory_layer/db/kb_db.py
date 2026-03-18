import faiss
import numpy as np
import os
import json
from typing import List, Dict, Any, Optional

class KBStore:
    """
    FAISS-based storage for the Knowledge Base.
    Handles vector indexing and metadata mapping with data integrity.
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

    def add(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        # Filtering to handle duplicates via content hashes
        new_indices = []
        existing_hashes = {m.get("content_hash") for m in self.metadata if "content_hash" in m}

        for i, meta in enumerate(metadatas):
            c_hash = meta.get("content_hash")
            if not c_hash or c_hash not in existing_hashes:
                new_indices.append(i)
                if c_hash:
                    existing_hashes.add(c_hash)

        if not new_indices:
            return

        filtered_embeddings = embeddings[new_indices]
        filtered_metadatas = [metadatas[i] for i in new_indices]

        faiss.normalize_L2(filtered_embeddings)
        self.index.add(filtered_embeddings)
        self.metadata.extend(filtered_metadatas)

        for i in range(filtered_embeddings.shape[0]):
            self.embeddings_cache.append(filtered_embeddings[i])
        self._save()

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                res = self.metadata[idx].copy()
                # Reinforcement: incorporate usage score if available
                usage_score = res.get("usage_score", 1.0)
                res["score"] = float(scores[0][i]) * usage_score
                results.append(res)
        return sorted(results, key=lambda x: x["score"], reverse=True)

    def record_success(self, chunk_id: str):
        """
        Feedback loop: Increment importance of successfully used chunks.
        """
        for meta in self.metadata:
            if meta["id"] == chunk_id:
                meta["usage_score"] = meta.get("usage_score", 1.0) + 0.1
                break
        self._save()

    def prune(self, min_score: float = 0.5):
        """
        Memory Pruning: remove stale, low-value chunks.
        """
        initial_count = len(self.metadata)
        indices_to_keep = [
            i for i, m in enumerate(self.metadata) if m.get("usage_score", 1.0) >= min_score
        ]

        if len(indices_to_keep) == initial_count:
            return

        self.metadata = [self.metadata[i] for i in indices_to_keep]
        self.embeddings_cache = [self.embeddings_cache[i] for i in indices_to_keep]
        self.index = faiss.IndexFlatIP(self.dimension)
        if self.embeddings_cache:
            self.index.add(np.array(self.embeddings_cache).astype('float32'))
        self._save()

    def clear_by_path(self, file_path: str):
        """
        Safely remove entries and rebuild the index to maintain data integrity.
        """
        indices_to_keep = [
            i for i, m in enumerate(self.metadata) if m.get("path") != file_path
        ]

        if len(indices_to_keep) == len(self.metadata):
            return

        # Rebuild metadata, cache and index
        new_metadata = [self.metadata[i] for i in indices_to_keep]
        new_cache = [self.embeddings_cache[i] for i in indices_to_keep]

        self.metadata = new_metadata
        self.embeddings_cache = new_cache
        self.index = faiss.IndexFlatIP(self.dimension)

        if self.embeddings_cache:
            embeddings_matrix = np.array(self.embeddings_cache).astype('float32')
            self.index.add(embeddings_matrix)

        self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            # We don't save the embeddings_cache to JSON (too large)
            # In a real system, we'd use a more robust storage like LanceDB or HDF5.
            json.dump(self.metadata, f, indent=2)

    def _load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            # Note: We can't easily reconstruct embeddings_cache from IndexFlatIP
            # because FAISS doesn't provide a public 'get_vectors' for all index types.
            # For this prototype, we'll populate it on first load if possible.
            if self.index.ntotal > 0:
                try:
                    # Flat index usually stores vectors in index.reconstruct_n
                    self.embeddings_cache = [self.index.reconstruct(i) for i in range(self.index.ntotal)]
                except:
                    pass
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
