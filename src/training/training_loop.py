# src/training/training_loop.py

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import torch
from typing import List, Dict


def tokenize_function(
    examples: Dict[str, List[str]], tokenizer: AutoTokenizer
) -> Dict[str, torch.Tensor]:
    """Tokenizes input and output fields for causal language modeling."""
    inputs = examples.get("input", [])
    outputs = examples.get("output", [])

    tokenized_inputs = tokenizer.batch_encode_plus(  # type: ignore[attr-defined]
        inputs,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    tokenized_outputs = tokenizer.batch_encode_plus(  # type: ignore[attr-defined]
        outputs,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )

    tokenized_inputs["labels"] = tokenized_outputs["input_ids"]
    return tokenized_inputs


def prepare_dataset(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    """Tokenizes the entire dataset for Trainer."""
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer), batched=True
    )
    return tokenized_dataset


def train_model(
    model_name: str,
    dataset: Dataset,
    output_dir: str = "./results",
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 5e-5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    tokenized_dataset = prepare_dataset(dataset, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=50,
        report_to=None,  # ❌ was "none", now None
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,  # type: ignore
    )

    trainer.train()
    return model, tokenizer
