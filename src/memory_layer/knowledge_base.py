# src/memory_layer/knowledge_base.py

import os
import json
import uuid
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class KnowledgeBase:
    """
    Long-term memory layer for Achilles Agent.
    Stores embeddings + metadata for persistent knowledge retrieval.
    """

    def __init__(
        self,
        storage_path: str = "backend/memory_layer/storage/knowledge_base.json",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.storage_path = storage_path
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)

        # Load or init store
        if os.path.exists(storage_path):
            with open(storage_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = {}

        # Embedding model
        self.model = SentenceTransformer(embedding_model)

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

        embeddings = self.model.encode(texts)
        ids = []

        for i, text in enumerate(texts):
            entry_id = str(uuid.uuid4())
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            self.data[entry_id] = {
                "text": text,
                "embedding": embeddings[i].tolist(),
                "metadata": metadata,
            }
            ids.append(entry_id)

        self._save()
        return ids

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant entries given a query string.
        Returns top_k entries sorted by cosine similarity using optimized numpy operations.
        """
        if not self.data:
            return []

        # 1. Generate query embedding
        query_embedding = self.model.encode([query])[0]

        # 2. Extract IDs and pre-calculated embeddings
        ids = list(self.data.keys())
        embeddings = np.array([v["embedding"] for v in self.data.values()])

        # 3. Optimized Vector Search (Cosine Similarity)
        # norm(A) * norm(B) * cos(theta) = A . B
        # If embeddings are normalized, it's just dot product.
        # Otherwise, use scikit-learn's cosine_similarity which is already optimized.
        query_vec = query_embedding.reshape(1, -1)
        scores = cosine_similarity(query_vec, embeddings)[0]

        # 4. Get top K results using argpartition for better performance on large sets
        if len(scores) > top_k:
            idx = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = idx[np.argsort(scores[idx])][::-1]
        else:
            top_indices = np.argsort(scores)[::-1]

        return [
            {
                "id": ids[i],
                "text": self.data[ids[i]]["text"],
                "metadata": self.data[ids[i]]["metadata"],
                "score": float(scores[i]),
            }
            for i in top_indices
        ]

    def update_entry(
        self, entry_id: str, new_text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update existing entry if present.
        Returns True if successful.
        """
        if entry_id not in self.data:
            return False

        embedding = self.model.encode([new_text])[0].tolist()
        self.data[entry_id].update(
            {
                "text": new_text,
                "embedding": embedding,
                "metadata": metadata or self.data[entry_id]["metadata"],
            }
        )
        self._save()
        return True

    def _save(self):
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)


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
