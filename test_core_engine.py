import os
import sys
import shutil
from storage import StorageManager
from indexer import run_indexer
from retrieval import retrieve
from feedback import log_event, RetrievalEvent

def test_pipeline():
    # 1. Setup temporary test directory and DB
    test_root = "test_workspace"
    db_path = "test_memory.db"

    if os.path.exists(test_root): shutil.rmtree(test_root)
    if os.path.exists(db_path): os.remove(db_path)
    os.makedirs(test_root)

    # Create some test files
    with open(os.path.join(test_root, "code.py"), "w") as f:
        f.write("def hello():\n    print('hello world')\n\nclass Master:\n    def run(self):\n        pass\n")

    with open(os.path.join(test_root, "readme.md"), "w") as f:
        f.write("# Project Alpha\n\nThis is a sample project for testing the cognitive memory engine.\n\nIt handles deterministic indexing.")

    db = StorageManager(db_path)

    print("--- 1. Initial Indexing ---")
    run_indexer(test_root, db)

    active_chunks = db.fetch_active_chunks()
    print(f"Active chunks in DB: {len(active_chunks)}")

    print("\n--- 2. Idempotency Check (Run Again) ---")
    run_indexer(test_root, db)
    print(f"Active chunks in DB (should be same): {len(db.fetch_active_chunks())}")

    print("\n--- 3. Retrieval Test ---")
    query = "How does the master run?"
    results = retrieve(query, db, top_k=2)
    print(f"Query: '{query}'")
    for r in results:
        print(f" - [{r['score']:.4f}] {r['chunk_id'][:8]}: {r['content'][:40].replace('\\n', ' ')}...")

    print("\n--- 4. Feedback Reinforcement ---")
    best_chunk_id = results[0]['chunk_id']
    print(f"Simulating selection of chunk: {best_chunk_id[:8]}")
    event = RetrievalEvent(query, [r['chunk_id'] for r in results], [best_chunk_id])
    log_event(event, db)

    # Check success_score
    cursor = db.conn.cursor()
    cursor.execute("SELECT success_score, retrieval_count FROM retrieval_stats WHERE chunk_id=?", (best_chunk_id,))
    score, count = cursor.fetchone()
    print(f"Updated Stats: score={score}, count={count}")

    print("\n--- 5. Incremental Indexing (File Change) ---")
    with open(os.path.join(test_root, "code.py"), "a") as f:
        f.write("\ndef extra():\n    return 42\n")

    run_indexer(test_root, db)
    print(f"Active chunks in DB (should increase): {len(db.fetch_active_chunks())}")

    print("\n--- 6. Deletion Handling ---")
    os.remove(os.path.join(test_root, "readme.md"))
    run_indexer(test_root, db)
    print(f"Active chunks in DB (should decrease): {len(db.fetch_active_chunks())}")

    db.close()
    # Cleanup
    shutil.rmtree(test_root)
    if os.path.exists(db_path): os.remove(db_path)
    print("\nPipeline Test Complete.")

if __name__ == "__main__":
    test_pipeline()
