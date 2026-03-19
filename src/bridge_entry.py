import sys
import json
import os
import threading
from UnifiedMemory import UnifiedMemory
from training.training_loop import train_model
from training.dataset_parser import DatasetParser

# Add core to path for core engine access
# Works in both dev (src/) and build (out/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Check sibling (for out/) and parent's sibling (for src/)
paths_to_check = [
    os.path.join(current_dir, "core"),
    os.path.join(os.path.dirname(current_dir), "core")
]
for p in paths_to_check:
    if os.path.exists(p):
        sys.path.append(p)
        break
from indexer import run_indexer
from datasets import Dataset

def main():
    # Robust path resolution
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    storage_dir = os.path.join(base_dir, "backend", "memory_layer", "storage")

    # Initialize once and keep in memory
    um = UnifiedMemory(
        os.path.join(storage_dir, "short_term_memory.db"),
        os.path.join(storage_dir, "quick_recall.json"),
        os.path.join(storage_dir, "knowledge_base.json")
    )

    print(json.dumps({"status": "ready"}), flush=True)

    # Simple stdin-based server loop
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            request_id = request.get("requestId")
            layer = request.get("layer")
            query = request.get("query", {})

            if layer == "kb":
                results = um.query_kb(query.get("text", ""), top_k=query.get("top_k", 5))
            elif layer == "kb_add":
                results = um.add_kb(query.get("text", ""), query.get("metadata", {}))
            elif layer == "kb_add_batch":
                results = um.add_kb_batch(query.get("texts", []), query.get("metadatas", []))
            elif layer == "kb_clear_file":
                results = um.clear_kb_file(query.get("path", ""))
            elif layer == "kb_rerank":
                results = um.rerank_kb(query.get("text", ""), query.get("results", []))
            elif layer == "kb_stats":
                results = um.kb.db.get_top_chunks(limit=query.get("limit", 10))
            elif layer == "feedback":
                event = RetrievalEvent(
                    query=query.get("query", ""),
                    retrieved_chunk_ids=query.get("retrieved_ids", []),
                    selected_chunk_ids=query.get("selected_ids", [])
                )
                log_event(event, um.kb.db)
                results = {"status": "success"}
            elif layer == "clear_all":
                results = um.clear_all()
            elif layer == "stm":
                results = um.query_short_term(query.get("key", ""), query.get("value", ""))
            elif layer == "stm_summarize":
                results = um.summarize_short_term(query.get("text", ""))
            elif layer == "stm_add":
                results = um.add_short_term(query.get("entry", {}))
            elif layer == "qr":
                results = um.query_quick_recall(query.get("embedding", []), query.get("top_k", 5))
            elif layer == "qr_add":
                results = um.add_quick_recall(query.get("entry", {}), query.get("embedding", None))
            elif layer == "run_indexer":
                root_path = query.get("root_path")
                file_path = query.get("file_path")
                if root_path:
                    run_indexer(root_path, um.kb.db)
                    results = {"status": "success", "message": f"Project indexed: {root_path}"}
                elif file_path:
                    # For single file, we wrap it as a scan of its parent but only processing it
                    # OR we could implement a more targeted function.
                    # For now, we reuse run_indexer logic if possible or just log.
                    run_indexer(os.path.dirname(file_path), um.kb.db)
                    results = {"status": "success", "message": f"File indexed: {file_path}"}
                else:
                    results = {"error": "No path provided for run_indexer"}
            elif layer == "train_on_kb":
                # Handle training in a separate thread to avoid blocking the bridge
                def background_training():
                    try:
                        parser = DatasetParser(um.kb.data)
                        instructions = parser.to_instruction_format()
                        ds = Dataset.from_list(instructions)
                        model_name = query.get("model", "sshleifer/tiny-gpt2")
                        train_model(model_name, ds, epochs=1)
                    except Exception as train_error:
                        print(f"DEBUG: Background training failed: {train_error}", file=sys.stderr)

                threading.Thread(target=background_training, daemon=True).start()
                results = {"status": "success", "message": "Training started in background"}
            else:
                results = {"error": f"Unknown layer: {layer}"}

            response = {"requestId": request_id, "data": results}
            print(json.dumps(response), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)

if __name__ == "__main__":
    main()
