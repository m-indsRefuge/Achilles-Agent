import sqlite3
from pathlib import Path

from achilles_memory.indexer import hash_file, index_project
from achilles_memory.retrieval import retrieve_chunks



def test_file_hash_changes_when_content_changes(tmp_path):
    file = tmp_path / "app.py"
    file.write_text("print('hello')")

    hash_one = hash_file(file)

    file.write_text("print('goodbye')")

    hash_two = hash_file(file)

    assert hash_one != hash_two



def test_indexing_same_project_twice_is_idempotent(tmp_path):
    project = tmp_path / "project"
    project.mkdir()

    app = project / "app.py"
    app.write_text("def hello():\n    return 'hi'")

    db_path = tmp_path / "memory.db"

    index_project(project, db_path)

    conn = sqlite3.connect(db_path)
    first_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    index_project(project, db_path)

    second_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    conn.close()

    assert first_count == second_count



def test_modified_file_marks_old_chunks_inactive(tmp_path):
    project = tmp_path / "project"
    project.mkdir()

    app = project / "app.py"
    app.write_text("version one")

    db_path = tmp_path / "memory.db"

    index_project(project, db_path)

    app.write_text("version two")

    index_project(project, db_path)

    conn = sqlite3.connect(db_path)

    inactive = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE is_active = 0"
    ).fetchone()[0]

    conn.close()

    assert inactive > 0



def test_deleted_file_marks_chunks_inactive(tmp_path):
    project = tmp_path / "project"
    project.mkdir()

    app = project / "app.py"
    app.write_text("temporary content")

    db_path = tmp_path / "memory.db"

    index_project(project, db_path)

    app.unlink()

    index_project(project, db_path)

    conn = sqlite3.connect(db_path)

    inactive = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE is_active = 0"
    ).fetchone()[0]

    conn.close()

    assert inactive > 0



def test_retrieval_returns_expected_chunk(tmp_path):
    project = tmp_path / "project"
    project.mkdir()

    app = project / "retry.py"
    app.write_text("Database connection retry logic")

    db_path = tmp_path / "memory.db"

    index_project(project, db_path)

    results = retrieve_chunks(db_path, "retry")

    assert any("retry logic" in result for result in results)
