# src/memory_layer/tests/test_integration.py

import os
import tempfile
import pytest

from memory_layer.knowledge_base import KnowledgeBase
from memory_layer.quick_recall import QuickRecall


@pytest.fixture
def temp_storage():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_end_to_end_integration(temp_storage):
    kb_path = os.path.join(temp_storage, "kb.json")
    qr_path = os.path.join(temp_storage, "qr.json")

    # Initialize components
    kb = KnowledgeBase(storage_path=kb_path)
    qr = QuickRecall(storage_path=qr_path)

    # Add entry to KB
    entry_id = kb.add_entry("Python factorial function", {"source": "unit_test"})
    assert entry_id in kb.data

    # Search KB
    results = kb.search("factorial")
    assert len(results) > 0
    top_result = results[0]
    assert "factorial" in top_result["text"].lower()

    # Add entry to QuickRecall using KB embedding
    embedding = kb.model.encode([top_result["text"]], convert_to_numpy=True)[0].tolist()
    qr.add(
        {
            "id": entry_id,
            "text": top_result["text"],
            "metadata": top_result["metadata"],
        },
        embedding=embedding,
    )

    # Query QuickRecall
    query_embedding = kb.model.encode(["factorial code"], convert_to_numpy=True)[0].tolist()
    recall_results = qr.query(query_embedding, top_k=1)

    assert len(recall_results) == 1
    assert "factorial" in recall_results[0]["text"].lower()