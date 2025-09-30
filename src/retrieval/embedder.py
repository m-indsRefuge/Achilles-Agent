# src/retrieval/embedder.py
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Simple embedder abstraction. Default uses sentence-transformers 'all-MiniLM-L6-v2'
    for low-memory CPU-friendly embeddings. Swap the model name to use EmbeddingGemma later.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = SentenceTransformer(model_name_or_path=model_name, device=device)

    def encode(
        self, texts: List[str], batch_size: int = 32, normalize: bool = True
    ) -> np.ndarray:
        """
        Encode a list of texts into numpy array shape (n, d).
        """
        emb = self._model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
        if normalize:
            # L2-normalize — useful for cosine / inner-product ranking consistency
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            emb = emb / norms
        return emb

    @property
    def dim(self) -> int:
        # dimension of the model used
        return self._model.get_sentence_embedding_dimension()
