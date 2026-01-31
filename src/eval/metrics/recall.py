import numpy as np


def recall_at_k(
    ranked_items: np.ndarray,
    true_ratings: dict[int, float],
    k: int = 10,
    threshold: float = 4.0,
) -> float:
    """Compute Recall@K with binary relevance.

    Parameters
    ----------
    ranked_items : array-like of int
        Item IDs ordered by predicted score (descending). Only the first
        *k* entries are used.
    true_ratings : dict[int, float]
        Mapping from item ID to its ground-truth rating (1–5). Items
        absent from this dict are treated as non-relevant.
    k : int
        Cut-off position.
    threshold : float
        Minimum rating to count as relevant.

    Returns
    -------
    float
        Recall@K in [0, 1]. Returns 0.0 when the user has no relevant items.
    """
    total_relevant = sum(1 for r in true_ratings.values() if r >= threshold)

    if total_relevant == 0:
        return 0.0

    ranked_items = np.asarray(ranked_items)[:k]
    hits = sum(1 for item in ranked_items if true_ratings.get(item, 0) >= threshold)
    return hits / total_relevant
