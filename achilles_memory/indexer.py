import hashlib
from pathlib import Path

from .storage import get_connection, initialize_database

SUPPORTED_EXTENSIONS = {".py", ".md", ".txt"}
CHUNK_SIZE = 20


def hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()



def hash_file(path: Path) -> str:
    return hash_content(path.read_text(encoding="utf-8"))



def chunk_text(content: str, chunk_size: int = CHUNK_SIZE):
    lines = content.splitlines()

    for index in range(0, len(lines), chunk_size):
        yield "\n".join(lines[index:index + chunk_size])



def chunk_id(file_path: str, chunk_index: int, content: str) -> str:
    raw = f"{file_path}:{chunk_index}:{hash_content(content)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()



def index_project(project_path: Path, db_path: Path):
    initialize_database(db_path)

    conn = get_connection(db_path)

    existing_paths = set()

    for file in project_path.rglob("*"):
        if not file.is_file():
            continue

        if file.suffix not in SUPPORTED_EXTENSIONS:
            continue

        relative_path = str(file.relative_to(project_path))
        existing_paths.add(relative_path)

        content = file.read_text(encoding="utf-8")
        file_hash = hash_file(file)

        existing = conn.execute(
            "SELECT id, hash FROM documents WHERE path = ?",
            (relative_path,),
        ).fetchone()

        if existing and existing["hash"] == file_hash:
            continue

        if existing:
            document_id = existing["id"]

            conn.execute(
                "UPDATE documents SET hash = CURRENT_TIMESTAMP WHERE id = ?",
                (document_id,),
            )

            conn.execute(
                "UPDATE chunks SET is_active = 0 WHERE document_id = ?",
                (document_id,),
            )
        else:
            cursor = conn.execute(
                "INSERT INTO documents(path, hash) VALUES(?, ?)",
                (relative_path, file_hash),
            )
            document_id = cursor.lastrowid

        for idx, chunk in enumerate(chunk_text(content)):
            cid = chunk_id(relative_path, idx, chunk)

            conn.execute(
                """
                INSERT OR REPLACE INTO chunks(
                    id,
                    document_id,
                    content,
                    content_hash,
                    chunk_index,
                    is_active
                ) VALUES (?, ?, ?, ?, ?, 1)
                """,
                (
                    cid,
                    document_id,
                    chunk,
                    hash_content(chunk),
                    idx,
                ),
            )

    rows = conn.execute("SELECT id, path FROM documents").fetchall()

    for row in rows:
        if row["path"] not in existing_paths:
            conn.execute(
                "UPDATE chunks SET is_active = 0 WHERE document_id = ?",
                (row["id"],),
            )

    conn.commit()
    conn.close()
