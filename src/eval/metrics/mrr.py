from typing import Dict, Tuple

import numpy as np

def mrr_at_k(
    ranked_item_ids: np.ndarray,
    true_ratings: Dict[int, float],
    k: int,
    threshold: float,
) -> Tuple[float, bool]:
    """
        Compute reciprocal rank at K for a single user.
    """
    relevant = {iid for iid, r in true_ratings.items() if r >= threshold}
    if not relevant:
        return 0.0, False

    topk = ranked_item_ids[:k]
    for rank, iid in enumerate(topk, start=1):
        if int(iid) in relevant:
            return 1.0 / rank, True
    return 0.0, True
