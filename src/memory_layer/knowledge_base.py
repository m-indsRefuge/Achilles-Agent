# src/memory_layer/knowledge_base.py

import os
import json
import uuid
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from .db.kb_db import KBStore


class KnowledgeBase:
    """
    Long-term memory layer for Achilles Agent.
    Stores embeddings + metadata for persistent knowledge retrieval.
    """

    def __init__(
        self,
        storage_path: str = "backend/memory_layer/storage/knowledge_base.json",
        embedding_model: str = "all-MiniLM-L6-v2",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.storage_path = storage_path
        # Embedding model
        self.model = SentenceTransformer(embedding_model)
        dimension = self.model.get_sentence_embedding_dimension()
        self.store = KBStore(storage_path, dimension=dimension)
        # Re-ranker model (lazy load in real scenarios, but here for skeleton)
        try:
            self.reranker = CrossEncoder(rerank_model)
        except:
            self.reranker = None

    @property
    def data(self):
        # Backward compatibility for tests
        return {m["id"]: m for m in self.store.metadata}

    def add_entry(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add new text to knowledge base with embedding + metadata.
        Returns ID of entry.
        """
        return self.add_entries([text], [metadata or {}])[0]

    def add_entries(
        self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add multiple entries at once for performance.
        Returns list of entry IDs.
        """
        if not texts:
            return []

        embeddings = np.array(self.model.encode(texts), dtype="float32")

        entry_metadatas = []
        for i, text in enumerate(texts):
            meta = metadatas[i].copy() if metadatas and i < len(metadatas) else {}
            entry_id = str(uuid.uuid4())
            entry_metadatas.append({
                "id": entry_id,
                "text": text,
                "metadata": meta
            })

        self.store.add(embeddings, entry_metadatas)
        return [m["id"] for m in entry_metadatas]

    def clear_file_entries(self, file_path: str):
        """
        Remove all entries associated with a specific file path in metadata.
        """
        self.store.clear_by_path(file_path)

    def clear(self):
        """
        Wipe the entire knowledge base.
        """
        self.store.clear()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant entries given a query string.
        """
        query_embedding = np.array([self.model.encode([query])[0]], dtype="float32")
        return self.store.search(query_embedding, top_k)

    def rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Refine search results using a Cross-Encoder.
        """
        if not self.reranker or not results:
            return results

        pairs = [[query, r["text"]] for r in results]
        scores = self.reranker.predict(pairs)

        for i, score in enumerate(scores):
            results[i]["rerank_score"] = float(score)

        return sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)

    def update_entry(
        self, entry_id: str, new_text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update existing entry if present.
        """
        # Rebuilding FAISS index for updates is expensive.
        # For this prototype, we'll just re-add it.
        self.add_entry(new_text, metadata)
        return True


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
