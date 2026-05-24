from pathlib import Path

from .storage import get_connection



def retrieve_chunks(db_path: Path, query: str, limit: int = 5):
    conn = get_connection(db_path)

    rows = conn.execute(
        """
        SELECT content
        FROM chunks
        WHERE is_active = 1
        AND LOWER(content) LIKE ?
        LIMIT ?
        """,
        (f"%{query.lower()}%", limit),
    ).fetchall()

    conn.close()

    return [row["content"] for row in rows]
