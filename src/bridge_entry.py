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
        # Standardized paths as used in the backend
        um = UnifiedMemory(
            "backend/memory_layer/storage/short_term_memory.json",
            "backend/memory_layer/storage/quick_recall.json",
            "backend/memory_layer/storage/knowledge_base.json"
        )

        if layer == "kb":
            results = um.query_kb(query.get("text", ""))
        elif layer == "stm":
            results = um.query_short_term(query.get("key", ""), query.get("value", ""))
        else:
            results = {"error": f"Unknown layer: {layer}"}

        print(json.dumps(results))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
