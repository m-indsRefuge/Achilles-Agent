import hashlib
from typing import List

# Default to a lightweight local model if available, otherwise use ASCII fallback
try:
    from sentence_transformers import SentenceTransformer
    # Deterministic model loading
    _MODEL_NAME = 'all-MiniLM-L6-v2'
    _model = SentenceTransformer(_MODEL_NAME)
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False

def embed_text(text: str) -> List[float]:
    """Deterministic text embedding."""
    if HAS_MODEL:
        # SentenceTransformers is deterministic by default
        return _model.encode(text).tolist()
    return _ascii_fallback_embedding(text)

def embed_query(text: str) -> List[float]:
    """Consistency across indexing and retrieval."""
    return embed_text(text)

def _ascii_fallback_embedding(text: str) -> List[float]:
    """ASCII fallback embedding (deterministic)."""
    h = hashlib.sha256(text.encode('utf-8')).digest()
    return [float(b) / 255.0 for b in h[:384]] # 384 dimensions matching MiniLM
