from typing import List, Dict
from memory_layer.MemoryManager import MemoryManager
from memory_layer.quick_recall import QuickRecall
from training.embedding_utils import generate_embedding


def memory_to_training_examples(
    kb: MemoryManager, qr: QuickRecall, query: str
) -> List[Dict]:
    """
    Pulls relevant examples from memory layers and prepares them
    as input-output pairs for training.
    """
    examples = []

    # Query KnowledgeBase
    kb_matches = kb.query(query)
    for entry in kb_matches:
        examples.append({"input": query, "output": entry["text"]})

    # Query QuickRecall
    qr_embedding = generate_embedding(query)
    qr_matches = qr.query(qr_embedding)
    for entry in qr_matches:
        examples.append({"input": query, "output": entry["text"]})

    return examples
