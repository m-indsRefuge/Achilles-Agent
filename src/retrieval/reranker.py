# src/retrieval/reranker.py
from typing import List, Dict, Any, Tuple
import numpy as np
from .embedder import Embedder
from .indexer import HnswIndexer
from crossencoder import CrossEncoder


class Reranker:
    """
    Two-stage retrieval + rerank:
      1) embed query
      2) ann search for top_n candidates
      3) re-score candidates with a CrossEncoder (query, candidate_text)
    Returns top_k ranked items with scores and metadata.
    """

    def __init__(
        self,
        embedder: Embedder,
        indexer: HnswIndexer,
        cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.embedder = embedder
        self.indexer = indexer
        self.cross = CrossEncoder(cross_encoder_name, device="cpu")

    def add_documents(self, docs: List[Dict[str, Any]]):
        """
        docs: list of {"id": int, "text": str, "meta": {...}}
        """
        ids = [int(d["id"]) for d in docs]
        texts = [d["text"] for d in docs]
        metas = [d.get("meta", {}) for d in docs]
        vectors = self.embedder.encode(texts)
        self.indexer.add(ids, vectors, metas)

    def search(
        self, query: str, top_n: int = 100, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        # 1) embed query
        qvec = self.embedder.encode([query])[0]
        # 2) ann search
        candidate_ids, dists = self.indexer.search(qvec, k=top_n)
        if len(candidate_ids) == 0:
            return []

        candidate_metas = self.indexer.get_meta(candidate_ids)
        candidate_texts = [m.get("text", "") for m in candidate_metas]

        # 3) cross-encoder re-score
        pairs = [[query, t] for t in candidate_texts]
        scores = self.cross.predict(pairs)  # higher = better
        # sort by score descending
        ranked = sorted(
            zip(candidate_ids, candidate_texts, candidate_metas, scores),
            key=lambda x: x[3],
            reverse=True,
        )
        results = []
        for _id, text, meta, score in ranked[:top_k]:
            r = {"id": int(_id), "text": text, "meta": meta, "score": float(score)}
            results.append(r)
        return results
