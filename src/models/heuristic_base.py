from __future__ import annotations

from abc import ABC
from typing import Dict, List

import numpy as np
import pandas as pd

from .base import Rating, RecommenderModel


class HeuristicRanker(RecommenderModel, ABC):
    """Base class for non-learned ranking heuristics."""

    def fit(
        self,
        ratings: pd.DataFrame,
        users: pd.DataFrame | None = None,
        movies: pd.DataFrame | None = None,
    ) -> "HeuristicRanker":
        """Fit heuristic-specific statistics."""
        raise NotImplementedError

    @staticmethod
    def build_seen_items(
        ratings: pd.DataFrame,
        user_col: str = "UserID",
        item_col: str = "MovieID",
    ) -> Dict[int, set[int]]:
        """Build map of seen items per user from interactions."""
        return {
            int(uid): set(group[item_col].astype(int).tolist())
            for uid, group in ratings.groupby(user_col)
        }

    @staticmethod
    def top_k_from_scores(
        item_ids: np.ndarray,
        scores: np.ndarray,
        k: int,
        seen: set[int] | None = None,
    ) -> List[Rating]:
        """Return top-k unseen items by score (descending)."""
        seen = seen or set()

        if item_ids.size == 0 or k <= 0:
            return []

        work_scores = scores.astype(np.float64, copy=True)
        if seen:
            seen_mask = np.isin(item_ids, np.fromiter(seen, dtype=item_ids.dtype))
            work_scores[seen_mask] = -np.inf

        valid_mask = np.isfinite(work_scores)
        valid_k = min(k, int(valid_mask.sum()))
        if valid_k == 0:
            return []

        top_idx = np.argpartition(-work_scores, valid_k - 1)[:valid_k]
        top_idx = top_idx[np.argsort(-work_scores[top_idx])]

        return [
            Rating(movie_id=int(item_ids[i]), score=float(scores[i]))
            for i in top_idx
            if np.isfinite(work_scores[i])
        ]
