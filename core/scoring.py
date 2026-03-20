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
            "similarity": 0.6,
            "feedback": 0.3,
            "recency": 0.1
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

        # 2. Feedback Component (Bounded function: s / (s + k))
        # Note: Success score itself should decay to prevent long-term bias
        success_score = chunk.get("success_score", 1.0)
        # Apply temporal decay to the feedback signal itself for true bias prevention
        decayed_success = success_score * math.exp(-self.decay_lambda * age_seconds)
        feedback_score = decayed_success / (decayed_success + self.feedback_k)

        # 3. Recency Component (Exponential Decay: exp(-lambda * age))
        recency_score = math.exp(-self.decay_lambda * age_seconds)

        # 4. Weighted Final Score
        final_score = (
            self.weights["similarity"] * similarity_score +
            self.weights["feedback"] * feedback_score +
            self.weights["recency"] * recency_score
        )

        return {
            "final_score": float(final_score),
            "components": {
                "similarity": float(similarity_score),
                "feedback": float(feedback_score),
                "recency": float(recency_score)
            }
        }
