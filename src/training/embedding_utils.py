from typing import List
import numpy as np


# Placeholder: integrate CodeLlama encoder or sentence-transformers later
def generate_embedding(text: str) -> List[float]:
    """
    Convert text to embedding vector.
    Currently a dummy example: replace with real encoder.
    """
    # For now, simple ASCII-based vector (placeholder)
    vec = np.zeros(128)
    for i, c in enumerate(text[:128]):
        vec[i] = ord(c) / 255.0
    return vec.tolist()
