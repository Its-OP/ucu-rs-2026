import numpy as np


def recall_at_k(
    ranked_items: np.ndarray,
    true_ratings: dict[int, float],
    k: int = 10,
    threshold: float = 4.0,
) -> float | None:
    """
        Recall@K with binary relevance.
    """
    total_relevant = sum(1 for r in true_ratings.values() if r >= threshold)

    if total_relevant == 0:
        return None

    effective_k = min(k, total_relevant)
    ranked_items = np.asarray(ranked_items)[:effective_k]
    hits = sum(1 for item in ranked_items if true_ratings.get(item, 0) >= threshold)
    return hits / total_relevant
