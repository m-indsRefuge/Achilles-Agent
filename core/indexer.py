import os
import hashlib
from typing import List, Dict, Any, Optional
from storage import StorageManager
from embedding import embed_text

class Chunk:
    def __init__(self, content: str, start_line: int, end_line: int, index: int):
        self.content = content
        self.start_line = start_line
        self.end_line = end_line
        self.index = index

def scan_directory(root_path: str) -> List[str]:
    """Recursively scan files ignoring specified patterns."""
    root_path = os.path.abspath(root_path)
    ignore_patterns = {'.git', '__pycache__'}
    file_list = []

    for root, dirs, files in os.walk(root_path):
        # Ignore directories in-place
        dirs[:] = [d for d in dirs if d not in ignore_patterns]

        for file in files:
            path = os.path.join(root, file)
            # Basic binary check
            if not is_binary(path):
                file_list.append(path)
    return file_list

def is_binary(path: str) -> bool:
    """Basic check for binary files."""
    try:
        with open(path, 'tr') as check_file:
            check_file.read(1024)
            return False
    except UnicodeDecodeError:
        return True

def compute_file_hash(path: str) -> str:
    """Compute SHA256 of a file in chunks."""
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def chunk_file(path: str) -> List[Chunk]:
    """Deterministic chunking based on file type."""
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception:
        return []

    ext = os.path.splitext(path)[1].lower()
    chunks = []

    if ext in {'.py', '.js', '.ts', '.java', '.cpp', '.c'}:
        # Boundary-aware chunking (functions, classes)
        current_chunk_lines = []
        start_line = 0
        chunk_idx = 0

        for i, line in enumerate(lines):
            # Deterministic boundaries
            is_boundary = line.startswith(('def ', 'class ', 'function ', 'export ', 'async def '))

            if (is_boundary or len(current_chunk_lines) >= 50) and current_chunk_lines:
                content = "".join(current_chunk_lines)
                chunks.append(Chunk(content, start_line + 1, i, chunk_idx))
                current_chunk_lines = []
                start_line = i
                chunk_idx += 1

            current_chunk_lines.append(line)

        if current_chunk_lines:
            content = "".join(current_chunk_lines)
            chunks.append(Chunk(content, start_line + 1, len(lines), chunk_idx))

    else:
        # Default: split by paragraphs
        content = "".join(lines)
        paragraphs = content.split('\n\n')
        line_offset = 0
        for i, p in enumerate(paragraphs):
            if p.strip():
                p_lines = p.count('\n') + 1
                chunks.append(Chunk(p, line_offset + 1, line_offset + p_lines, i))
                line_offset += p_lines + 1 # +1 for the \n\n split
            else:
                line_offset += 1

    return chunks

def normalize_content(content: str) -> str:
    """Ensures content canonicalization before hashing and storage."""
    if not content:
        return ""
    content = content.strip().replace("\r\n", "\n")
    return " ".join(content.split())

def compute_chunk_id(file_path: str, chunk_index: int, content: str) -> str:
    """SHA256(file_path + chunk_index + normalized_content)."""
    normalized_content = normalize_content(content)
    raw = f"{file_path}{chunk_index}{normalized_content}"
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def run_indexer(root_path: str, db: StorageManager):
    """The indexing pipeline."""
    all_files = scan_directory(root_path)

    # Track existing documents to handle deletions
    db_cursor = db.conn.cursor()
    db_cursor.execute("SELECT path, id FROM documents")
    existing_docs = {row[0]: row[1] for row in db_cursor.fetchall()}
    processed_paths = set()

    for path in all_files:
        current_hash = compute_file_hash(path)
        existing_doc = db.get_document_by_path(path)
        processed_paths.add(path)

        if existing_doc and existing_doc['content_hash'] == current_hash:
            continue

        # Changed or New: re-chunk
        if existing_doc:
            db.deactivate_chunks_for_document(existing_doc['id'])

        doc_id = db.upsert_document(path, current_hash)

        chunks = chunk_file(path)
        chunk_data_list = []
        embedding_list = []

        for c in chunks:
            chunk_id = compute_chunk_id(path, c.index, c.content)
            normalized_content = normalize_content(c.content)
            chunk_data_list.append({
                'id': chunk_id,
                'document_id': doc_id,
                'content': c.content,
                'content_hash': hashlib.sha256(normalized_content.encode('utf-8')).hexdigest(),
                'start_line': c.start_line,
                'end_line': c.end_line
            })

            # Embed
            vector = embed_text(c.content)
            import array
            vector_blob = array.array('f', vector).tobytes()
            embedding_list.append((chunk_id, vector_blob))

        if chunk_data_list:
            db.insert_chunks(chunk_data_list)
            db.insert_embeddings(embedding_list)

    # Handle deletions
    for path, doc_id in existing_docs.items():
        if path not in processed_paths:
            db.deactivate_chunks_for_document(doc_id)
