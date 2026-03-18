import os
import hashlib
from typing import List, Dict, Any

class Chunk:
    def __init__(self, content: str, start_line: int, end_line: int, index: int):
        self.content = content
        self.start_line = start_line
        self.end_line = end_line
        self.index = index


def scan_directory(root_path: str) -> List[str]:
    file_paths = []
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in [".git", "__pycache__"]]
        for file in files:
            full_path = os.path.abspath(os.path.join(root, file))
            if is_binary(full_path):
                continue
            file_paths.append(full_path)
    return file_paths


def is_binary(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(1024)
            return b"\0" in chunk
    except:
        return True


def compute_file_hash(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def compute_chunk_id(file_path: str, chunk_index: int, content: str) -> str:
    normalized = content.strip()
    raw = f"{file_path}:{chunk_index}:{normalized}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def chunk_file(path: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return chunks

    chunk_size = 50
    for i in range(0, len(lines), chunk_size):
        content = "".join(lines[i:i + chunk_size])
        chunks.append(Chunk(content, i, min(i + chunk_size, len(lines)), i // chunk_size))
    return chunks


def embed_text(text: str) -> List[float]:
    return [float(ord(c)) for c in text[:128]]


def run_indexer(root_path: str, db) -> None:
    file_paths = scan_directory(root_path)
    existing_paths = set(file_paths)

    for path in file_paths:
        file_hash = compute_file_hash(path)
        doc = db.get_document_by_path(path)

        if doc and doc["content_hash"] == file_hash:
            continue

        document_id = db.upsert_document(path, file_hash)
        db.deactivate_chunks_for_document(document_id)

        chunks = chunk_file(path)

        for chunk in chunks:
            chunk_id = compute_chunk_id(path, chunk.index, chunk.content)

            db.insert_chunk({
                "id": chunk_id,
                "document_id": document_id,
                "content": chunk.content,
                "content_hash": hashlib.sha256(chunk.content.encode()).hexdigest(),
                "start_line": chunk.start_line,
                "end_line": chunk.end_line
            })

            embedding = embed_text(chunk.content)
            db.insert_embedding(chunk_id, embedding)

    all_docs = db.get_all_documents()
    for doc in all_docs:
        if doc["path"] not in existing_paths:
            db.deactivate_chunks_for_document(doc["id"])
