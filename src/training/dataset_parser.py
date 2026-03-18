import json
from typing import List, Dict

class DatasetParser:
    """
    Parses Knowledge Base entries into training data formats for LLM fine-tuning.
    """
    def __init__(self, kb_data: Dict):
        self.kb_data = kb_data

    def to_instruction_format(self) -> List[Dict[str, str]]:
        """
        Converts indexed code/text into (instruction, input, output) format.
        """
        dataset = []
        for entry_id, entry in self.kb_data.items():
            text = entry.get("text", "")
            if not text:
                continue

            # Simple heuristic for training data generation
            dataset.append({
                "instruction": "Explain or complete the following code snippet.",
                "input": text[:200], # First 200 chars as input
                "output": text
            })
        return dataset

    def save_jsonl(self, output_path: str):
        data = self.to_instruction_format()
        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
