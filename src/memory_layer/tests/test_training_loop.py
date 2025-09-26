# src/memory_layer/tests/test_training_loop.py

import pytest
from datasets import Dataset
from training.training_loop import train_model


def test_training_loop():
    # Minimal test dataset
    test_data = [
        {"input": "print hello world", "output": "print('hello world')"},
        {"input": "add numbers 2 and 3", "output": "2 + 3"},
    ]
    test_dataset = Dataset.from_list(test_data)

    # Run training for 1 epoch with batch size 1
    model_name = "sshleifer/tiny-gpt2"
    model, tokenizer = train_model(
        model_name=model_name, dataset=test_dataset, epochs=1, batch_size=1
    )

    # Simple sanity check
    assert model is not None
    assert tokenizer is not None
    print("Test training loop completed successfully.")
