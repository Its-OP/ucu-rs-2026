import numpy as np


def ndcg_at_k(ranked_items: np.ndarray, true_ratings: dict[int, float], k: int = 10) -> float:
    """Compute NDCG@K with graded relevance.

    Parameters
    ----------
    ranked_items : np.ndarray
        Item IDs ordered by predicted score (descending). Only the first
        *k* entries are used.
    true_ratings : dict[int, float]
        Mapping from item ID to its ground-truth rating (1–5). Items
        absent from this dict are treated as having zero gain.
    k : int
        Cut-off position.

    Returns
    -------
    float
        NDCG@K in [0, 1]. Returns 0.0 when there are no rated items.
    """
    ranked_items = np.asarray(ranked_items)[:k]
    gains = np.array([true_ratings.get(item, 0.0) for item in ranked_items])
    discounts = np.log2(np.arange(2, len(gains) + 2))  # log2(i+1) for i=1..K

    dcg = np.sum(gains / discounts)

    ideal_gains = np.sort(list(true_ratings.values()))[::-1][:k]
    ideal_discounts = np.log2(np.arange(2, len(ideal_gains) + 2))
    idcg = np.sum(ideal_gains / ideal_discounts)

    if idcg == 0.0:
        return 0.0

    return float(dcg / idcg)
