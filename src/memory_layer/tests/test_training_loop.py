# src/memory_layer/tests/test_training_loop.py

import time
import tracemalloc
import torch
from datasets import Dataset
from training.training_loop import train_model


def test_training_loop():
    """Integration test: ensure training runs end-to-end."""
    test_data = [
        {"input": "print hello world", "output": "print('hello world')"},
        {"input": "add numbers 2 and 3", "output": "2 + 3"},
    ]
    dataset = Dataset.from_list(test_data)

    model, tokenizer = train_model(
        model_name="sshleifer/tiny-gpt2",
        dataset=dataset,
        epochs=1,
        batch_size=1,
    )

    assert model is not None
    assert tokenizer is not None
    assert tokenizer.pad_token is not None


def test_pad_token_setup():
    """Unit test: pad token should be set if missing."""
    dataset = Dataset.from_list([{"input": "x", "output": "y"}])

    _, tokenizer = train_model(
        model_name="sshleifer/tiny-gpt2",
        dataset=dataset,
        epochs=1,
        batch_size=1,
    )

    assert tokenizer.pad_token is not None
    assert tokenizer.pad_token_id is not None


def test_training_loop_stress_with_memory():
    """Stress test: run on larger synthetic dataset."""
    test_data = [{"input": f"input {i}", "output": f"output {i}"} for i in range(200)]
    dataset = Dataset.from_list(test_data)

    tracemalloc.start()
    start_time = time.time()

    model, tokenizer = train_model(
        model_name="sshleifer/tiny-gpt2",
        dataset=dataset,
        epochs=2,
        batch_size=4,
    )

    duration = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert model is not None
    assert tokenizer is not None
    assert duration < 120  # sanity check runtime
    assert peak < 1e9  # sanity check memory (<1GB)
