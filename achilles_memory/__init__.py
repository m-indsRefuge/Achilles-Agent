"""Achilles Agent memory layer.

This package contains the first deterministic, testable memory-layer primitives:
file hashing, chunking, SQLite-backed storage, indexing, and lightweight retrieval.
"""

from .indexer import index_project
from .retrieval import retrieve_chunks

__all__ = ["index_project", "retrieve_chunks"]
