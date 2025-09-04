import os
import json
from collections import deque
from typing import Any, Deque, Dict, Optional


class ShortTermMemory:
    def __init__(self, max_size: int = 50, storage_path: Optional[str] = None):
        """
        Short-term memory for quick recall of recent items.

        :param max_size: Maximum number of items to retain in memory.
        :param storage_path: Optional path to persist memory between sessions.
        """
        self.max_size: int = max_size
        self.storage_path: Optional[str] = storage_path
        self.memory: Deque[Dict[str, Any]] = deque(maxlen=max_size)

        # Load memory if storage_path exists
        self._load()

    def add(self, item: Dict[str, Any]) -> None:
        """
        Add an item to short-term memory and persist if storage_path is set.

        :param item: The item to store.
        """
        self.memory.append(item)
        self._persist()

    def query(self, key: str, value: Any) -> Optional[Dict[str, Any]]:
        """
        Query memory for an item matching key=value.

        :param key: Dictionary key to match.
        :param value: Value to match.
        :return: The first matching item or None if not found.
        """
        for item in reversed(self.memory):
            if key in item and item[key] == value:
                return item
        return None

    def _persist(self) -> None:
        """
        Save memory to disk if storage_path is set.
        """
        if self.storage_path is not None:
            try:
                with open(self.storage_path, "w", encoding="utf-8") as f:
                    json.dump(list(self.memory), f, indent=2)
            except Exception as e:
                print(f"Error persisting memory: {e}")

    def _load(self) -> None:
        """
        Load memory from disk if storage_path exists.
        """
        if self.storage_path is not None and os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    items = json.load(f)
                    self.memory = deque(items, maxlen=self.max_size)
            except Exception:
                # If corrupted or unreadable, start fresh
                self.memory = deque(maxlen=self.max_size)

    def clear(self) -> None:
        """
        Clear the memory and delete persisted file if exists.
        """
        self.memory.clear()
        if self.storage_path is not None and os.path.exists(self.storage_path):
            try:
                os.remove(self.storage_path)
            except Exception as e:
                print(f"Error removing memory file: {e}")
