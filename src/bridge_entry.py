import sys
import json
import os
from UnifiedMemory import UnifiedMemory

def main():
    # Robust path resolution
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    storage_dir = os.path.join(base_dir, "backend", "memory_layer", "storage")

    # Initialize once and keep in memory
    um = UnifiedMemory(
        os.path.join(storage_dir, "short_term_memory.json"),
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
            elif layer == "stm":
                results = um.query_short_term(query.get("key", ""), query.get("value", ""))
            elif layer == "stm_add":
                results = um.add_short_term(query.get("entry", {}))
            elif layer == "qr":
                results = um.query_quick_recall(query.get("embedding", []), query.get("top_k", 5))
            elif layer == "qr_add":
                results = um.add_quick_recall(query.get("entry", {}), query.get("embedding", None))
            else:
                results = {"error": f"Unknown layer: {layer}"}

            response = {"requestId": request_id, "data": results}
            print(json.dumps(response), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)

if __name__ == "__main__":
    main()
