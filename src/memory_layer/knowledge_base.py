# backend/memory_layer/knowledge_base.py

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
        entry_id = str(uuid.uuid4())
        embedding = self.model.encode([text])[0].tolist()
        self.data[entry_id] = {
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {},
        }
        self._save()
        return entry_id

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant entries given a query string.
        Returns top_k entries sorted by cosine similarity.
        """
        if not self.data:
            return []

        query_embedding = self.model.encode([query])[0].reshape(1, -1)
        ids, embeddings = zip(*[(k, v["embedding"]) for k, v in self.data.items()])
        embeddings = np.array(embeddings)

        scores = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = scores.argsort()[-top_k:][::-1]

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
