# src/memory_layer/knowledge_base.py

import os
import json
import uuid
import hashlib
import logging
import array
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import sys
# Works in both dev (src/) and build (out/)
current_dir = os.path.dirname(os.path.abspath(__file__))
paths_to_check = [
    os.path.join(os.path.dirname(current_dir), "core"),
    os.path.join(os.path.dirname(os.path.dirname(current_dir)), "core")
]
for p in paths_to_check:
    if os.path.exists(p):
        sys.path.append(p)
        break
from storage import StorageManager
from retrieval import retrieve as core_retrieve
from indexer import run_indexer


class KnowledgeBase:
    """
    Long-term memory layer for Achilles Agent.
    Stores embeddings + metadata for persistent knowledge retrieval.
    """

    def __init__(
        self,
        storage_path: str = "backend/memory_layer/storage/knowledge_base.sqlite",
        embedding_model: str = "all-MiniLM-L6-v2",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.logger = logging.getLogger("KnowledgeBase")
        self.storage_path = storage_path
        # Embedding model
        self.model = SentenceTransformer(embedding_model)

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

        # Migration from legacy usage_score if knowledge_base.json exists
        # Done before persistent DB connection to avoid lock issues during migration
        self._migrate_legacy_data()

        # Core engine DB connection (persistent for extension lifecycle)
        self.db = StorageManager(self.storage_path)

        # Re-ranker model (lazy load)
        self.rerank_model_name = rerank_model
        self._reranker = None

    def _migrate_legacy_data(self):
        legacy_path = self.storage_path.replace(".sqlite", ".json")
        if not os.path.exists(legacy_path):
            return

        try:
            self.logger.info(f"Migrating legacy data from {legacy_path}")
            with open(legacy_path, 'r', encoding='utf-8') as f:
                legacy_data = json.load(f)

            # Open temporary SQLite connection for migration
            temp_db = StorageManager(self.storage_path)

            # Map usage_score -> success_score with scale factor 10
            for entry in legacy_data:
                u_score = entry.get("usage_score", 1.0)
                s_score = u_score * 10.0

                meta = entry.get("metadata", {})
                path = meta.get("path", f"legacy_{entry.get('id')}")

                doc_id = temp_db.upsert_document(path, "legacy_migration")

                # We can't easily recover embeddings from JSON,
                # but we ensure the stats exist for the ID
                chunk_data = {
                    'id': entry.get('id', str(uuid.uuid4())),
                    'document_id': doc_id,
                    'content': entry.get('text', ''),
                    'content_hash': hashlib.sha256(entry.get('text', '').encode('utf-8')).hexdigest(),
                    'start_line': meta.get('lineStart', 1),
                    'end_line': meta.get('lineStart', 1) + entry.get('text', '').count('\n')
                }
                temp_db.insert_chunk(chunk_data)

                # Directly update migrated score
                cursor = temp_db.conn.cursor()
                cursor.execute(
                    "UPDATE retrieval_stats SET success_score = ? WHERE chunk_id = ?",
                    (s_score, chunk_data['id'])
                )
                temp_db.conn.commit()

            temp_db.close()
            # Rename legacy file to prevent re-migration
            os.rename(legacy_path, legacy_path + ".bak")
            self.logger.info("Legacy migration complete.")
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")

    @property
    def reranker(self):
        if self._reranker is None:
            try:
                self._reranker = CrossEncoder(self.rerank_model_name)
            except Exception as e:
                self.logger.error(f"Failed to load reranker: {e}")
        return self._reranker

    def __del__(self):
        if hasattr(self, 'db'):
            try:
                self.db.close()
            except:
                pass

    @property
    def data(self):
        # Backward compatibility for tests - Fetch active chunks from core engine
        # Ensures all data retrieval uses the unified engine
        chunks = self.db.fetch_active_chunks()
        return {c["id"]: {
            "id": c["id"],
            "text": c["content"],
            "success_score": c.get("success_score", 1.0),
            "last_updated": c.get("last_updated")
        } for c in chunks}

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
        Now also populates the deterministic core engine.
        """
        if not texts:
            return []

        embeddings = np.array(self.model.encode(texts), dtype="float32")

        entry_metadatas = []
        chunk_data_list = []
        embedding_list = []

        for i, text in enumerate(texts):
            meta = metadatas[i].copy() if metadatas and i < len(metadatas) else {}
            entry_id = str(uuid.uuid4())
            entry_metadatas.append({
                "id": entry_id,
                "text": text,
                "metadata": meta
            })

            # Sync with core engine
            doc_id = self.db.upsert_document(meta.get("path", f"kb_{entry_id}"), "manual")
            chunk_data_list.append({
                'id': entry_id,
                'document_id': doc_id,
                'content': text,
                'content_hash': hashlib.sha256(text.encode('utf-8')).hexdigest(),
                'start_line': meta.get("lineStart", 1),
                'end_line': meta.get("lineStart", 1) + text.count('\n')
            })
            embedding_list.append((entry_id, array.array('f', embeddings[i]).tobytes()))

        if chunk_data_list:
            self.db.insert_chunks(chunk_data_list)
            self.db.insert_embeddings(embedding_list)

        return [m["id"] for m in entry_metadatas]

    def clear_file_entries(self, file_path: str):
        """
        Remove all entries associated with a specific file path in metadata.
        """
        doc = self.db.get_document_by_path(file_path)
        if doc:
            self.db.deactivate_chunks_for_document(doc['id'])

    def clear(self):
        """
        Wipe the entire knowledge base.
        """
        # Re-initialize DB tables (simple wipe for prototype)
        if os.path.exists(self.storage_path):
             self.db.close()
             os.remove(self.storage_path)
             self.db = StorageManager(self.storage_path)

    def search(
        self, query: str, top_k: int = 5, task_type: Optional[str] = None, expand_context: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Delegates search to the core engine. This is the single source of truth for retrieval.
        """
        try:
            # If we're performing a search for user UI or project analysis,
            # we want the full multi-hop/context expansion features
            if expand_context:
                results = core_retrieve(query, self.db, top_k=top_k)
            else:
                from retrieval import retrieve_no_event
                results = retrieve_no_event(query, self.db, top_k=top_k)

            # Re-formatting for the extension expected output (mapping keys)
            formatted = []
            for r in results:
                # Core results already use RetrievalScorer.score()
                formatted.append({
                    "id": r["chunk_id"],
                    "text": r.get("context", r.get("content", "")),
                    "score": r["score"], # Contains {final, components}
                    "path": r.get("source_path") or r.get("path"),
                    "hop": r.get("hop", 1),
                    "metadata": {
                        "path": r.get("source_path") or r.get("path"),
                        "success_score": r.get("success_score", 1.0)
                    }
                })

            # Apply re-ranking if CrossEncoder is available
            if self.reranker and formatted:
                 return self.rerank(query, formatted)

            return formatted
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    def rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Refine search results using a Cross-Encoder.
        """
        model = self.reranker
        if not model or not results:
            return results

        pairs = [[query, r["text"]] for r in results]
        scores = model.predict(pairs)

        for i, score in enumerate(scores):
            results[i]["rerank_score"] = float(score)

        return sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)

    def update_entry(
        self, entry_id: str, new_text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update existing entry if present.
        """
        # For the new engine, we just re-add it (idempotent upsert by ID)
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
