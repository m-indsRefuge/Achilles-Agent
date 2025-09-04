from memory_layer.MemoryManager import MemoryManager
from memory_layer.quick_recall import QuickRecall
from memory_layer.short_term_memory import ShortTermMemory
from training.dataset_utils import memory_to_training_examples
from training.embedding_utils import generate_embedding

# Paths
kb_path = "data/knowledge_base.json"
qr_path = "data/quick_recall.json"
stm_path = "data/short_term_memory.json"

# Initialize memory layers
kb = MemoryManager(storage_path=kb_path)
qr = QuickRecall(storage_path=qr_path)
stm = ShortTermMemory(max_size=50, storage_path=stm_path)


def train_model(query: str):
    """
    Example training loop:
    - Pulls examples from memory layers
    - Feeds into CodeLlama fine-tuning pipeline
    """
    examples = memory_to_training_examples(kb, qr, query)
    print(f"Training on {len(examples)} examples for query: {query}")

    # Placeholder: integrate CodeLlama fine-tuning here
    for ex in examples:
        # ex["input"] -> prompt
        # ex["output"] -> expected completion
        pass


if __name__ == "__main__":
    test_query = "Python factorial function"
    train_model(test_query)
