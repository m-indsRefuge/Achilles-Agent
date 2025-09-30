# src/training/training_loop.py

from typing import Any, Dict, Tuple, cast
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)

# short max length to keep tests fast and memory-light
MAX_LENGTH = 128
DEFAULT_LR = 5e-5


def tokenize_function(
    examples: Dict[str, Any], tokenizer: PreTrainedTokenizerBase
) -> Dict[str, Any]:
    """
    Tokenize a batch (examples contains lists for 'input' and 'output').
    Returns a mapping with lists for input_ids, attention_mask, and labels.
    Uses tokenizer(...) (the modern __call__) - compatible with type checkers.
    """
    inputs = examples["input"]
    outputs = examples["output"]

    tokenized_inputs = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    tokenized_outputs = tokenizer(
        outputs,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs.get("attention_mask"),
        "labels": tokenized_outputs["input_ids"],
    }


def prepare_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    """
    Tokenize the Hugging Face Dataset, remove other columns, and set PyTorch format.
    """
    tokenized = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer), batched=True
    )

    # Keep only tensor-friendly columns
    keep = [
        c
        for c in tokenized.column_names
        if c in ("input_ids", "attention_mask", "labels")
    ]
    tokenized = tokenized.remove_columns(
        [c for c in tokenized.column_names if c not in keep]
    )

    # Set dataset to return PyTorch tensors
    tokenized.set_format(type="torch", columns=keep)
    return tokenized


def train_model(
    model_name: str,
    dataset: Dataset,
    epochs: int = 1,
    batch_size: int = 2,
    lr: float = DEFAULT_LR,
    device: torch.device | None = None,
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
    """
    Minimal training loop using PyTorch directly (no Trainer / accelerate).
    - Loads tokenizer + model
    - Ensures pad_token exists (adds '[PAD]' if necessary) and resizes embeddings
    - Tokenizes the dataset and converts to PyTorch tensors
    - Runs a small training loop and returns (model, tokenizer)
    """

    # Load tokenizer and model with strict casting for Pylance
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(model_name),
    )
    model = cast(
        AutoModelForCausalLM,
        AutoModelForCausalLM.from_pretrained(model_name),
    )

    # Ensure pad token exists (GPT-2 family typically doesn't have one)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    # Prepare dataset
    tokenized_dataset = prepare_dataset(dataset, tokenizer)

    # Device selection
    device = device or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    model.to(device)

    # DataLoader
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for _ in range(max(1, int(epochs))):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["labels"].to(device)

            kwargs: Dict[str, Any] = {"input_ids": input_ids, "labels": labels}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask

            outputs = model(**kwargs)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

    return model, tokenizer
