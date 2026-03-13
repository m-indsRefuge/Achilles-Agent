import sys
import json
import os
from UnifiedMemory import UnifiedMemory

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Missing arguments"}))
        sys.exit(1)

    layer = sys.argv[1]
    query_str = sys.argv[2]

    try:
        query = json.loads(query_str)
        # Robust path resolution
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        storage_dir = os.path.join(base_dir, "backend", "memory_layer", "storage")

        um = UnifiedMemory(
            os.path.join(storage_dir, "short_term_memory.json"),
            os.path.join(storage_dir, "quick_recall.json"),
            os.path.join(storage_dir, "knowledge_base.json")
        )

        if layer == "kb":
            results = um.query_kb(query.get("text", ""))
        elif layer == "kb_add":
            results = um.add_kb(query.get("text", ""), query.get("metadata", {}))
        elif layer == "kb_add_batch":
            results = um.add_kb_batch(query.get("texts", []), query.get("metadatas", []))
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

        print(json.dumps(results))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
