import json
from typing import List, Dict

class DatasetParser:
    """
    Parses Knowledge Base entries into training data formats for LLM fine-tuning.
    Designed for memory efficiency in constrained (4GB RAM) environments.
    """
    def __init__(self, kb_data_iterator):
        # Allow passing an iterator/generator to avoid holding all chunks in list
        self.kb_data = kb_data_iterator

    def to_instruction_format(self, max_samples: int = 1000):
        """
        Generator for (instruction, input, output) format to save memory.
        """
        count = 0
        # Check if kb_data is a dict (legacy) or items
        items = self.kb_data.items() if hasattr(self.kb_data, 'items') else self.kb_data

        for entry_id, entry in items:
            if count >= max_samples:
                break

            text = entry.get("text", "")
            if not text or len(text) < 50:
                continue

            # Efficient sampling of training pairs
            yield {
                "instruction": "Explain or complete the following code snippet.",
                "input": text[:150],
                "output": text[:1024] # Limit output size for training memory
            }
            count += 1

    def save_jsonl(self, output_path: str):
        data = self.to_instruction_format()
        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
