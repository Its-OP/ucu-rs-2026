from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd

from src.models.base import Rating
from src.models.heuristic_base import HeuristicRanker


class PopularityBase(HeuristicRanker, ABC):
    """Base class for global popularity-style rankers."""

    def __init__(self) -> None:
        self._item_ids: np.ndarray = np.array([], dtype=np.int64)
        self._scores: np.ndarray = np.array([], dtype=np.float64)

    def fit(
        self,
        ratings: pd.DataFrame,
        users: pd.DataFrame | None = None,
        movies: pd.DataFrame | None = None,
    ) -> "PopularityBase":
        """Fit item popularity statistics from training interactions.

        Parameters
        ----------
        ratings : pd.DataFrame
            Observed interactions (UserID, MovieID, Rating, Timestamp).
        users : pd.DataFrame | None
            User side-information. Not required for popularity statistics.
        movies : pd.DataFrame | None
            Movie side-information. Not required for popularity statistics.

        Returns
        -------
        PopularityBase
            Fitted ranker with global item scores sorted in descending order.
        """
        if ratings.empty:
            self._item_ids = np.array([], dtype=np.int64)
            self._scores = np.array([], dtype=np.float64)
            return self

        work = ratings[["MovieID", "Rating"]].copy()
        work["weight"] = self._compute_weights(ratings)
        work["weighted_rating"] = work["Rating"] * work["weight"]

        grouped = work.groupby("MovieID", sort=False)
        weighted_count = grouped["weight"].sum().astype(float)
        weighted_sum = grouped["weighted_rating"].sum().astype(float)
        weighted_mean = weighted_sum / weighted_count

        score_series = self._score_items(
            weighted_count=weighted_count,
            weighted_mean=weighted_mean,
            work=work,
        )

        ranking = pd.DataFrame({
            "MovieID": score_series.index.astype(int),
            "score": score_series.values.astype(float),
            "count_tiebreak": weighted_count.reindex(score_series.index).values.astype(
                float
            ),
        }).sort_values(
            by=["score", "count_tiebreak", "MovieID"],
            ascending=[False, False, True],
        )

        self._item_ids = ranking["MovieID"].to_numpy(dtype=np.int64)
        self._scores = ranking["score"].to_numpy(dtype=np.float64)
        return self

    def predict(
        self,
        users: pd.DataFrame,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        k: int = 10,
    ) -> Dict[int, List[Rating]]:
        """Produce top-K recommendations for each user.

        Parameters
        ----------
        users : pd.DataFrame
            User side-information (UserID, Gender, Age, Occupation, Zip-code).
        ratings : pd.DataFrame
            Observed interactions (UserID, MovieID, Rating, Timestamp).
        movies : pd.DataFrame
            Movie side-information (movie_id, title, genres).
        k : int
            Number of recommendations per user.

        Returns
        -------
        dict[int, list[Rating]]
            Mapping from UserID to a list of Rating objects sorted by
            score descending (length up to k).
        """
        seen_by_user = self.build_seen_items(ratings)
        candidate_ids, candidate_scores = self._catalog_scores(movies)

        preds: Dict[int, List[Rating]] = {}
        for uid in users["UserID"].astype(int).values:
            preds[int(uid)] = self.top_k_from_scores(
                item_ids=candidate_ids,
                scores=candidate_scores,
                k=k,
                seen=seen_by_user.get(int(uid), set()),
            )

        return preds

    def _catalog_scores(self, movies: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Restrict scores to movie catalog passed at prediction time."""
        if self._item_ids.size == 0:
            return self._item_ids, self._scores

        if "MovieID" in movies.columns:
            catalog_ids = movies["MovieID"].to_numpy(dtype=np.int64)
        elif "movie_id" in movies.columns:
            catalog_ids = movies["movie_id"].to_numpy(dtype=np.int64)
        else:
            return self._item_ids, self._scores

        mask = np.isin(self._item_ids, catalog_ids)
        return self._item_ids[mask], self._scores[mask]

    def _compute_weights(self, ratings: pd.DataFrame) -> np.ndarray:
        """Return per-interaction weight used for aggregated popularity stats."""
        return np.ones(len(ratings), dtype=np.float64)

    @abstractmethod
    def _score_items(
        self,
        weighted_count: pd.Series,
        weighted_mean: pd.Series,
        work: pd.DataFrame,
    ) -> pd.Series:
        """Compute final per-item score from aggregated statistics."""
        raise NotImplementedError


class PopularityRanker(PopularityBase):
    """Rank items by (possibly weighted) interaction count."""

    def _score_items(
        self,
        weighted_count: pd.Series,
        weighted_mean: pd.Series,
        work: pd.DataFrame,
    ) -> pd.Series:
        del weighted_mean, work
        return weighted_count.astype(float)


class BayesianPopularityRanker(PopularityBase):
    """Rank items by Bayesian-smoothed weighted mean rating."""

    def __init__(self, bayesian_m: float = 25.0) -> None:
        super().__init__()
        self.bayesian_m = float(bayesian_m)

    def _score_items(
        self,
        weighted_count: pd.Series,
        weighted_mean: pd.Series,
        work: pd.DataFrame,
    ) -> pd.Series:
        global_mean = float(
            np.sum(work["Rating"].to_numpy() * work["weight"].to_numpy())
            / np.sum(work["weight"].to_numpy())
        )
        v = weighted_count.astype(float)
        return (v / (v + self.bayesian_m)) * weighted_mean + (
            self.bayesian_m / (v + self.bayesian_m)
        ) * global_mean


class RecencyPopularityRanker(PopularityBase):
    """Rank items by recency-decayed interaction count."""

    def __init__(self, half_life_days: float = 30.0) -> None:
        super().__init__()
        self.half_life_days = float(half_life_days)

    def _compute_weights(self, ratings: pd.DataFrame) -> np.ndarray:
        if "Timestamp" not in ratings.columns:
            return np.ones(len(ratings), dtype=np.float64)

        ts = pd.to_datetime(ratings["Timestamp"])
        max_ts = ts.max()
        age_days = (max_ts - ts).dt.total_seconds().to_numpy() / 86400.0
        return np.exp(-np.log(2.0) * age_days / self.half_life_days)

    def _score_items(
        self,
        weighted_count: pd.Series,
        weighted_mean: pd.Series,
        work: pd.DataFrame,
    ) -> pd.Series:
        del weighted_mean, work
        return weighted_count.astype(float)
