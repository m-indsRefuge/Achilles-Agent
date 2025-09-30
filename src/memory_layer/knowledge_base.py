# src/memory_layer/Knowledge_Base.py

import os
import json
import uuid
from typing import List, Dict, Any, Optional, Union

import numpy as np


class DummyEmbedder:
    """Fallback embedder if none is provided (for CPU-only setups)."""

    def encode(self, texts: List[str]) -> np.ndarray:
        # Return random vectors just so the pipeline works
        return np.random.rand(len(texts), 384).astype(np.float32)


class DummyIndexer:
    """Simple in-memory indexer for prototyping."""

    def __init__(self):
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []

    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]):
        self.vectors.extend(vectors)
        self.metadata.extend(metadata)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.vectors:
            return []

        sims = np.dot(self.vectors, query_vec.T).flatten()
        top_idx = sims.argsort()[::-1][:top_k]

        return [self.metadata[i] for i in top_idx]


class Reranker:
    """Placeholder reranker until proper model is integrated."""

    def __init__(self, model: Optional[Any] = None):
        self.model = model

    def rank(
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        # TODO: Implement proper reranking with model
        # For now, just return candidates unchanged
        return candidates


class KnowledgeBase:
    def __init__(
        self,
        embedder: Optional[Any] = None,
        indexer: Optional[Any] = None,
        reranker: Optional[Any] = None,
    ):
        self.embedder = embedder or DummyEmbedder()
        self.indexer = indexer or DummyIndexer()
        self.reranker = reranker or Reranker()

    def _ensure_list(self, texts: Union[str, List[str]]) -> List[str]:
        return [texts] if isinstance(texts, str) else texts

    def add_document(
        self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None
    ):
        text_list = self._ensure_list(text)
        vector = self.embedder.encode(text_list)
        self.indexer.add(vector, [metadata or {"id": doc_id, "text": text}])

    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_vec = self.embedder.encode(self._ensure_list(query))
        candidates = self.indexer.search(query_vec[0], top_k=top_k)
        return self.reranker.rank(query, candidates)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Alias for query(), kept for backward compatibility."""
        return self.query(query, top_k=top_k)
