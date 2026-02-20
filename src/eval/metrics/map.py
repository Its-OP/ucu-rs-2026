from typing import Dict

import numpy as np

def map_at_k(
    ranked_item_ids: np.ndarray,
    true_ratings: Dict[int, float],
    k: int,
    threshold: float,
) -> float:
    """
        Compute Average Precision at K for a single user.
    """
    relevant = {iid for iid, r in true_ratings.items() if r >= threshold}
    if not relevant:
        return 0.0
    
    topk = ranked_item_ids[:k]
    hits = 0
    ap = 0.0
    for rank, iid in enumerate(topk, start=1):
        if int(iid) in relevant:
            hits += 1
            ap += hits / rank
    denom = min(len(relevant), k)
    if denom <= 0:
        return 0.0
    return ap / denom