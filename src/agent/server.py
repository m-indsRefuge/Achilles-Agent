# src/agent/server.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from memory_layer.knowledge_base import KnowledgeBase

app = FastAPI(title="Achilles Agent (dev)")

kb = KnowledgeBase()  # uses default reranker / embedder / indexer


class QueryRequest(BaseModel):
    q: str
    top_n: int = 100
    top_k: int = 5


@app.post("/query")
def query(req: QueryRequest):
    results = kb.search(req.q, top_n=req.top_n, top_k=req.top_k)
    return {"query": req.q, "results": results}
