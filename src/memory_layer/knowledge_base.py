# backend/memory_layer/knowledge_base.py

import os
import json
import uuid
from typing import List, Dict, Any, Optional, Iterable, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


class KnowledgeBase:
    """
    Long-term memory layer for Achilles Agent.
    Stores embeddings + metadata for persistent knowledge retrieval.

    Improvements over the original version:
    - Safe directory creation and robust JSON loading
    - Atomic file saves to avoid partial writes
    - Normalized embeddings with dot-product scoring (no sklearn dependency)
    - In-memory index for faster search and optional metadata filtering
    - Helper methods for batch add, get, delete, and length
    """

    def __init__(
        self,
        storage_path: str = "backend/memory_layer/storage/knowledge_base.json",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.storage_path = storage_path
        dirpath = os.path.dirname(storage_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        # Embedding model (normalize for cosine via dot product)
        self.model = SentenceTransformer(embedding_model)

        # Load or init store
        self.data: Dict[str, Dict[str, Any]] = self._load_data()

        # Build in-memory index for fast search
        self._ids: List[str] = []
        self._emb_matrix: Optional[np.ndarray] = None
        self._rebuild_index()

    # --------------- Public API ---------------
    def add_entry(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add new text to knowledge base with embedding + metadata.
        Returns ID of entry.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        entry_id = str(uuid.uuid4())
        embedding = self._encode_text(text)
        self.data[entry_id] = {
            "text": text,
            "embedding": embedding.tolist(),
            "metadata": metadata or {},
        }
        self._append_to_index(entry_id, embedding)
        self._save()
        return entry_id

    def add_entries(
        self,
        texts: Iterable[str],
        metadatas: Optional[Iterable[Optional[Dict[str, Any]]]] = None,
    ) -> List[str]:
        """
        Batch add multiple texts. Returns list of new entry IDs.
        """
        texts_list = list(texts)
        if metadatas is None:
            metadatas_list = [None] * len(texts_list)
        else:
            metadatas_list = list(metadatas)
            if len(metadatas_list) != len(texts_list):
                raise ValueError("texts and metadatas must be the same length")

        # Encode in batch for throughput
        embeddings = self._encode_batch(texts_list)
        new_ids: List[str] = []
        for text, meta, emb in zip(texts_list, metadatas_list, embeddings):
            if not isinstance(text, str) or not text.strip():
                continue
            entry_id = str(uuid.uuid4())
            self.data[entry_id] = {
                "text": text,
                "embedding": emb.tolist(),
                "metadata": meta or {},
            }
            new_ids.append(entry_id)

        # Update index in one go
        self._rebuild_index()
        self._save()
        return new_ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant entries given a query string.
        Returns up to top_k entries sorted by similarity.

        - where: optional exact-match metadata filters, e.g., {"namespace": "v1"}
        - min_score: optional minimum similarity threshold (after normalization)
        """
        if not self.data:
            return []

        top_k = int(top_k)
        if top_k <= 0:
            return []

        query_emb = self._encode_text(query)

        # Select candidates (optionally filter by metadata)
        if where:
            candidate_pairs: List[Tuple[int, str]] = [
                (i, eid)
                for i, eid in enumerate(self._ids)
                if self._entry_matches_where(self.data[eid].get("metadata", {}), where)
            ]
            if not candidate_pairs:
                return []
            cand_indices = np.array([i for i, _ in candidate_pairs], dtype=int)
            cand_ids = [eid for _, eid in candidate_pairs]
            cand_matrix = self._emb_matrix[cand_indices]
        else:
            cand_ids = self._ids
            cand_matrix = self._emb_matrix

        # Dot-product since all vectors are normalized -> cosine similarity
        scores = cand_matrix @ query_emb

        # Get top_k indices
        k = min(top_k, scores.shape[0])
        top_indices = np.argpartition(scores, -k)[-k:]
        # Sort descending
        top_sorted = top_indices[np.argsort(scores[top_indices])[::-1]]

        results: List[Dict[str, Any]] = []
        for idx in top_sorted:
            eid = cand_ids[int(idx)]
            score = float(scores[int(idx)])
            if min_score is not None and score < float(min_score):
                continue
            item = self.data[eid]
            results.append(
                {
                    "id": eid,
                    "text": item["text"],
                    "metadata": item.get("metadata", {}),
                    "score": score,
                }
            )

        return results

    def update_entry(
        self, entry_id: str, new_text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update existing entry if present.
        Returns True if successful.
        """
        if entry_id not in self.data:
            return False

        embedding = self._encode_text(new_text)
        # Shallow-merge metadata if provided
        current_meta = self.data[entry_id].get("metadata", {})
        new_meta = {**current_meta, **(metadata or {})}

        self.data[entry_id].update(
            {
                "text": new_text,
                "embedding": embedding.tolist(),
                "metadata": new_meta,
            }
        )
        self._update_index_row(entry_id, embedding)
        self._save()
        return True

    def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Return the entry dict by id or None if not found."""
        return self.data.get(entry_id)

    def delete_entry(self, entry_id: str) -> bool:
        """Delete an entry by id. Returns True if deleted."""
        if entry_id not in self.data:
            return False
        del self.data[entry_id]
        self._rebuild_index()
        self._save()
        return True

    def __len__(self) -> int:
        return len(self.data)

    # --------------- Internal utilities ---------------
    def _entry_matches_where(self, metadata: Dict[str, Any], where: Dict[str, Any]) -> bool:
        for key, expected in where.items():
            if metadata.get(key) != expected:
                return False
        return True

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode and normalize a single text as float32 numpy array."""
        emb = self.model.encode(
            [text], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        if emb.dtype != np.float32:
            emb = emb.astype(np.float32)
        return emb

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode and normalize a batch of texts as float32 numpy matrix."""
        embs = self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        if embs.dtype != np.float32:
            embs = embs.astype(np.float32)
        return embs

    def _load_data(self) -> Dict[str, Dict[str, Any]]:
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if not isinstance(loaded, dict):
                    return {}
                # Ensure required keys and types; best-effort normalization on load
                for eid, item in list(loaded.items()):
                    text = item.get("text", "")
                    emb = item.get("embedding")
                    meta = item.get("metadata") or {}
                    if not isinstance(text, str) or not text:
                        # Drop invalid entries
                        loaded.pop(eid, None)
                        continue
                    if not isinstance(emb, list):
                        # Recompute missing/invalid embeddings
                        emb_np = self._encode_text(text)
                        loaded[eid] = {"text": text, "embedding": emb_np.tolist(), "metadata": meta}
                return loaded
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _save(self, data: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Atomically save JSON to storage_path."""
        to_write = data if data is not None else self.data
        tmp_path = f"{self.storage_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(to_write, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.storage_path)

    def _rebuild_index(self) -> None:
        self._ids = list(self.data.keys())
        if not self._ids:
            self._emb_matrix = np.zeros((0, 1), dtype=np.float32)
            return
        matrix = np.array(
            [self.data[eid]["embedding"] for eid in self._ids], dtype=np.float32
        )
        # Normalize in case existing data was stored unnormalized
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        nonzero = norms.squeeze() > 0
        if np.any(nonzero):
            matrix[nonzero] = matrix[nonzero] / norms[nonzero]
        self._emb_matrix = matrix

    def _append_to_index(self, entry_id: str, embedding: np.ndarray) -> None:
        if self._emb_matrix is None or self._emb_matrix.size == 0:
            self._ids = [entry_id]
            self._emb_matrix = embedding.reshape(1, -1).astype(np.float32)
        else:
            self._ids.append(entry_id)
            self._emb_matrix = np.vstack([self._emb_matrix, embedding.astype(np.float32)])

    def _update_index_row(self, entry_id: str, embedding: np.ndarray) -> None:
        try:
            idx = self._ids.index(entry_id)
        except ValueError:
            # Not in index; rebuild
            self._rebuild_index()
            return
        self._emb_matrix[idx] = embedding.astype(np.float32)


if __name__ == "__main__":
    kb = KnowledgeBase()

    # Test run
    print("Adding entry...")
    entry_id = kb.add_entry(
        "Python function to compute factorial.", {"source": "manual_test"}
    )
    print("Entry added:", entry_id)

    print("\nSearching for 'factorial code'...")
    results = kb.search("factorial code")
    for r in results:
        print(r)
