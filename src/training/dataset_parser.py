# src/training/dataset_parser.py

from typing import List, Dict
from datasets import Dataset, load_dataset as hf_load_dataset
import csv
import json
import os


def load_huggingface_dataset(name: str, split: str = "train") -> Dataset:
    """
    Load a Hugging Face dataset with a given split.
    Always returns a `Dataset`, never a DatasetDict.
    """
    try:
        dataset = hf_load_dataset(name, split=split)
        if not isinstance(dataset, Dataset):
            raise TypeError(
                f"Expected Hugging Face Dataset, got {type(dataset)}. "
                f"Make sure 'split' is provided."
            )
        return dataset
    except Exception as e:
        raise RuntimeError(f"Failed to load Hugging Face dataset '{name}': {e}") from e


def load_csv_dataset(path: str) -> List[Dict]:
    """
    Load a dataset from a CSV file into a list of dicts.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def load_json_dataset(path: str) -> List[Dict]:
    """
    Load a dataset from a JSON file (list of dicts expected).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of objects")
        return data


def parse_dataset(
    source: str, source_type: str = "csv", split: str = "train"
) -> Dataset:
    """
    Parse dataset from different sources and always return a Hugging Face Dataset.

    Args:
        source: Hugging Face dataset name or file path.
        source_type: "huggingface", "csv", or "json".
        split: Split to use if loading from Hugging Face.

    Returns:
        Dataset: A Hugging Face Dataset object.
    """
    source_type = source_type.lower()

    if source_type == "huggingface":
        return load_huggingface_dataset(source, split=split)

    elif source_type == "csv":
        rows = load_csv_dataset(source)
        return Dataset.from_list(rows)

    elif source_type == "json":
        rows = load_json_dataset(source)
        return Dataset.from_list(rows)

    else:
        raise ValueError(f"Unsupported source_type: {source_type}")
