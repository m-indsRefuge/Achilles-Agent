import math
import time
from typing import Dict, List, Optional

class RetrievalScorer:
    """
    Structured, explainable, and extensible scoring engine for retrieval.
    Single source of truth for normalized ranking scores.
    """
    def __init__(self,
                 weights: Optional[Dict[str, float]] = None,
                 decay_lambda: float = 1e-6,
                 feedback_k: float = 5.0):
        self.weights = weights or {
            "similarity": 0.7,
            "feedback": 0.3
        }
        self.decay_lambda = decay_lambda
        self.feedback_k = feedback_k

    def score(self, chunk: Dict, query_embedding: List[float], metadata: Optional[Dict] = None) -> Dict:
        """
        Calculates normalized score components and weighted final score.
        """
        # 1. Similarity Component (Cosine Similarity normalized to [0, 1])
        # Proper normalization: (raw + 1.0) / 2.0 preserves full distribution
        raw_similarity = metadata.get("raw_similarity", 0.0) if metadata else 0.0
        similarity_score = (raw_similarity + 1.0) / 2.0
        similarity_score = max(0.0, min(1.0, similarity_score))

        current_time = time.time()
        last_time_str = chunk.get('last_updated') or chunk.get('last_accessed') or chunk.get('created_at')

        last_time = current_time
        if isinstance(last_time_str, (int, float)):
             last_time = last_time_str
        elif last_time_str:
            try:
                import datetime
                if 'T' in last_time_str:
                    last_time = datetime.datetime.fromisoformat(last_time_str).timestamp()
                else:
                    last_time = datetime.datetime.strptime(last_time_str, '%Y-%m-%d %H:%M:%S').timestamp()
            except (ValueError, TypeError):
                last_time = current_time

        age_seconds = max(0, current_time - last_time)

        # 2. Feedback Component (Direct normalized signal)
        feedback_score = chunk.get("success_score", 0.1)
        feedback_score = max(0.0, min(1.0, feedback_score))

        # 3. Recency Component (Exponential Decay: exp(-lambda * age))
        # Deterministic recency: cap age to ensure floating point stability across environments
        age_seconds = min(age_seconds, 31536000) # 1 year max age
        recency_score = math.exp(-self.decay_lambda * age_seconds)

        # 4. Weighted Final Score with Floating Point Stabilization
        # Uses dynamic weights if provided in metadata
        s_weight = metadata.get("similarity_weight", self.weights.get("similarity", 0.7)) if metadata else self.weights.get("similarity", 0.7)
        f_weight = metadata.get("feedback_weight", self.weights.get("feedback", 0.3)) if metadata else self.weights.get("feedback", 0.3)

        final_score = (s_weight * similarity_score) + (f_weight * feedback_score)

        # Round components for determinism
        return {
            "final_score": round(float(final_score), 6),
            "components": {
                "similarity": round(float(similarity_score), 6),
                "feedback": round(float(feedback_score), 6),
                "recency": round(float(recency_score), 6)
            }
        }
