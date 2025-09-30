# src/retrieval/tests/test_reranker_recall.py
import tempfile, os
from retrieval.embedder import Embedder
from retrieval.indexer import HnswIndexer
from retrieval.reranker import Reranker


def test_reranker_basic_recall():
    embedder = Embedder()
    idx_path = os.path.join(tempfile.mkdtemp(), "rr_idx")
    indexer = HnswIndexer(dim=embedder.dim, index_path=idx_path, max_elements=1000)
    rer = Reranker(embedder, indexer)
    # add small corpus
    docs = [
        {"id": 1, "text": "how to open a file in python", "meta": {"source": "doc1"}},
        {
            "id": 2,
            "text": "connecting to postgres using psycopg2",
            "meta": {"source": "doc2"},
        },
        {"id": 3, "text": "sorting a list in python", "meta": {"source": "doc3"}},
    ]
    rer.add_documents(docs)
    res = rer.search("open file read mode python", top_n=10, top_k=2)
    assert len(res) >= 1
    # top result should be doc 1 (or at least include text about opening files)
    texts = [r["text"] for r in res]
    assert any("open a file" in t.lower() or "open file" in t.lower() for t in texts)
