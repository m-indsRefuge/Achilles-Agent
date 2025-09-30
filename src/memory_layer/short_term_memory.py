import json
import os
from typing import Any, Dict, List, Optional


class ShortTermMemory:
    def __init__(self, max_size: int = 3, storage_path: str = "short_term_memory.json"):
        self.max_size = max_size
        self.storage_path = storage_path
        self.memory: List[Dict[str, Any]] = []
        self.load()

    def add(self, item: Dict[str, Any]) -> None:
        """Add a new memory item, evict oldest if size exceeded."""
        self.memory.append(item)
        if len(self.memory) > self.max_size:
            self.memory.pop(0)
        self.save()

    def query(self, key: str, value: Any) -> Optional[Dict[str, Any]]:
        """
        Query memory for an entry where the given key contains the given value.
        Handles both string and non-string fields by converting to str.
        """
        matches = [
            item
            for item in self.memory
            if key in item and str(value).lower() in str(item[key]).lower()
        ]
        return matches[0] if matches else None

    def clear(self) -> None:
        """Clear all memory items and delete storage file."""
        self.memory.clear()
        if os.path.exists(self.storage_path):
            try:
                os.remove(self.storage_path)
            except Exception as e:
                print(f"Error deleting STM file: {e}")

    def save(self) -> None:
        """Persist memory to file."""
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            print(f"Error saving STM: {e}")

    def load(self) -> None:
        """Load memory from file if exists."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.memory = json.load(f)
            except Exception as e:
                print(f"Error loading STM: {e}")
                self.memory = []
