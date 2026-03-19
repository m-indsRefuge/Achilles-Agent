import os
import sys
import shutil
from storage import StorageManager
from indexer import run_indexer
from retrieval import retrieve
from feedback import log_event, RetrievalEvent

def test_evolution_pipeline():
    # 1. Setup temporary test directory and DB
    test_root = "test_evolution_workspace"
    db_path = "test_evolution.db"

    if os.path.exists(test_root): shutil.rmtree(test_root)
    if os.path.exists(db_path): os.remove(db_path)
    os.makedirs(test_root)

    # Create test files with potential entities and context
    with open(os.path.join(test_root, "logic.py"), "w") as f:
        f.write("""
class Processor:
    def process_data(self, data):
        return data.upper()

def run_main():
    p = Processor()
    print(p.process_data('hello'))
""")

    with open(os.path.join(test_root, "readme.md"), "w") as f:
        f.write("# Evolution Project\n\nThis project tests multi-hop retrieval.\n\nIt mentions Processor class.")

    db = StorageManager(db_path)

    print("--- 1. Indexing ---")
    run_indexer(test_root, db)

    active_chunks = db.fetch_active_chunks()
    print(f"Active chunks: {len(active_chunks)}")

    print("\n--- 2. Multi-Hop Retrieval & Context Expansion ---")
    # Query should trigger keyword match on 'multi-hop' or 'Processor'
    query = "Tell me about multi-hop and Processor class"
    results = retrieve(query, db, top_k=2)

    print(f"Query: '{query}'")
    for r in results:
        print(f" - [{r['score']:.4f}] Hop: {r['hop']} Chunk: {r['chunk_id'][:8]} Path: {r['source_path']}")
        print(f"   Context:\n{r['context']}")
        print("-" * 20)

    # 3. Observability Check
    print("\n--- 3. Observability Check ---")
    cursor = db.conn.cursor()
    cursor.execute("SELECT query, retrieved_chunk_ids FROM retrieval_events")
    event = cursor.fetchone()
    if event:
        print(f"Logged Event Query: {event[0]}")
        print(f"Logged Chunk IDs: {event[1][:50]}...")

    top_chunks = db.get_top_chunks(limit=5)
    print(f"Top chunks: {top_chunks}")

    db.close()
    # Cleanup
    shutil.rmtree(test_root)
    if os.path.exists(db_path): os.remove(db_path)
    print("\nEvolution Pipeline Test Complete.")

if __name__ == "__main__":
    test_evolution_pipeline()
