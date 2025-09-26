# src/memory_layer/tests/test_memory_layer_extended.py

import pytest
from memory_layer.knowledge_base import KnowledgeBase
from memory_layer.quick_recall import QuickRecall
import os
import tempfile


@pytest.fixture
def temp_storage():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_add_and_search_kb(temp_storage):
    kb_path = os.path.join(temp_storage, "kb.json")
    kb = KnowledgeBase(storage_path=kb_path)

    entry_id = kb.add_entry("Python is a programming language", {"source": "unit_test"})
    assert entry_id in kb.data

    results = kb.search("programming")
    assert len(results) > 0
    assert "Python" in results[0]["text"]


def test_update_kb_entry(temp_storage):
    kb_path = os.path.join(temp_storage, "kb.json")
    kb = KnowledgeBase(storage_path=kb_path)

    entry_id = kb.add_entry("Initial text")
    updated = kb.update_entry(entry_id, "Updated text")
    assert updated is True

    results = kb.search("Updated")
    assert any("Updated" in r["text"] for r in results)


def test_quick_recall_add_and_query(temp_storage):
    qr_path = os.path.join(temp_storage, "qr.json")
    qr = QuickRecall(storage_path=qr_path)

    qr.add({"text": "Hello world"}, embedding=[0.1, 0.2, 0.3])
    results = qr.query([0.1, 0.2, 0.3])
    assert len(results) > 0
    assert results[0]["text"] == "Hello world"


def test_quick_recall_clear(temp_storage):
    qr_path = os.path.join(temp_storage, "qr.json")
    qr = QuickRecall(storage_path=qr_path)

    qr.add({"text": "Temp entry"}, embedding=[0.1, 0.2, 0.3])
    qr.clear()
    assert qr.memory == []
    assert not os.path.exists(qr_path)
