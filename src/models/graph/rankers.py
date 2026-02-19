from __future__ import annotations

from abc import ABC
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import sparse

from src.models.base import Rating
from src.models.heuristic_base import HeuristicRanker


class GraphRankerBase(HeuristicRanker, ABC):
    """Base class for graph-based item rankers."""

    def __init__(
        self,
        relevance_threshold: float = 4.0,
        use_rating_weights: bool = False,
    ) -> None:
        self.relevance_threshold = float(relevance_threshold)
        self.use_rating_weights = bool(use_rating_weights)

        self.item_ids: np.ndarray = np.array([], dtype=np.int64)
        self._item_to_idx: Dict[int, int] = {}
        self._transition: sparse.csr_matrix | None = None
        self._global_prior: np.ndarray = np.array([], dtype=np.float64)

    def fit(
        self,
        ratings: pd.DataFrame,
        users: pd.DataFrame | None = None,
        movies: pd.DataFrame | None = None,
    ) -> "GraphRankerBase":
        """Build item-item transition graph from training interactions.

        Parameters
        ----------
        ratings : pd.DataFrame
            Observed interactions (UserID, MovieID, Rating, Timestamp).
        users : pd.DataFrame | None
            User side-information. Not required to build graph topology.
        movies : pd.DataFrame | None
            Movie side-information. If provided, defines item catalog.

        Returns
        -------
        GraphRankerBase
            Fitted ranker with transition matrix and item index mapping.
        """
        if movies is not None and not movies.empty:
            if "MovieID" in movies.columns:
                self.item_ids = movies["MovieID"].drop_duplicates().astype(int).to_numpy()
            elif "movie_id" in movies.columns:
                self.item_ids = movies["movie_id"].drop_duplicates().astype(int).to_numpy()
            else:
                self.item_ids = (
                    ratings["MovieID"].drop_duplicates().astype(int).to_numpy()
                    if not ratings.empty
                    else np.array([], dtype=np.int64)
                )
        else:
            self.item_ids = (
                ratings["MovieID"].drop_duplicates().astype(int).to_numpy()
                if not ratings.empty
                else np.array([], dtype=np.int64)
            )

        self._item_to_idx = {mid: i for i, mid in enumerate(self.item_ids.tolist())}

        if ratings.empty or self.item_ids.size == 0:
            self._transition = sparse.csr_matrix((0, 0), dtype=np.float64)
            self._global_prior = np.array([], dtype=np.float64)
            return self

        filtered = ratings[ratings["MovieID"].isin(self._item_to_idx)].copy()
        filtered = filtered[filtered["Rating"] >= self.relevance_threshold].copy()

        if filtered.empty:
            n_items = len(self.item_ids)
            self._transition = sparse.csr_matrix((n_items, n_items), dtype=np.float64)
            self._global_prior = np.full(n_items, 1.0 / n_items, dtype=np.float64)
            return self

        users_list = filtered["UserID"].drop_duplicates().astype(int).to_numpy()
        user_to_idx = {uid: i for i, uid in enumerate(users_list.tolist())}
        rows = filtered["UserID"].map(user_to_idx).to_numpy(dtype=np.int64)
        cols = filtered["MovieID"].map(self._item_to_idx).to_numpy(dtype=np.int64)

        if self.use_rating_weights:
            vals = filtered["Rating"].to_numpy(dtype=np.float64)
        else:
            vals = np.ones(len(filtered), dtype=np.float64)

        user_item = sparse.csr_matrix(
            (vals, (rows, cols)),
            shape=(len(users_list), len(self.item_ids)),
            dtype=np.float64,
        )

        item_item = (user_item.T @ user_item).tocsr()
        item_item.setdiag(0.0)
        item_item.eliminate_zeros()

        row_sum = np.asarray(item_item.sum(axis=1)).ravel()
        nonzero = row_sum > 0

        if np.any(nonzero):
            inv = np.zeros_like(row_sum, dtype=np.float64)
            inv[nonzero] = 1.0 / row_sum[nonzero]
            d_inv = sparse.diags(inv)
            self._transition = (d_inv @ item_item).tocsr()
            self._global_prior = row_sum.astype(np.float64)
            self._global_prior /= self._global_prior.sum()
        else:
            n_items = len(self.item_ids)
            self._transition = sparse.csr_matrix((n_items, n_items), dtype=np.float64)
            self._global_prior = np.full(n_items, 1.0 / n_items, dtype=np.float64)

        return self

    def _catalog_mask(self, movies: pd.DataFrame) -> np.ndarray:
        if self.item_ids.size == 0:
            return np.array([], dtype=bool)

        if "MovieID" in movies.columns:
            catalog_ids = movies["MovieID"].to_numpy(dtype=np.int64)
        elif "movie_id" in movies.columns:
            catalog_ids = movies["movie_id"].to_numpy(dtype=np.int64)
        else:
            return np.ones(self.item_ids.shape[0], dtype=bool)

        return np.isin(self.item_ids, catalog_ids)

    def _user_positive_sets(self, ratings: pd.DataFrame) -> Dict[int, set[int]]:
        if ratings.empty:
            return {}
        pos = ratings[ratings["Rating"] >= self.relevance_threshold]
        return {
            int(uid): set(group["MovieID"].astype(int).tolist())
            for uid, group in pos.groupby("UserID")
        }

    def _seed_vector(self, positive_items: set[int]) -> np.ndarray:
        n_items = len(self.item_ids)
        if n_items == 0:
            return np.array([], dtype=np.float64)

        if not positive_items:
            return self._global_prior.copy()

        idx = [self._item_to_idx[mid] for mid in positive_items if mid in self._item_to_idx]
        if not idx:
            return self._global_prior.copy()

        seed = np.zeros(n_items, dtype=np.float64)
        seed[np.array(idx, dtype=np.int64)] = 1.0
        seed /= seed.sum()
        return seed


class ItemGraphPropagationRanker(GraphRankerBase):
    """Multi-step item-graph propagation from user positive seeds."""

    def __init__(
        self,
        relevance_threshold: float = 4.0,
        use_rating_weights: bool = False,
        alpha: float = 0.85,
        n_steps: int = 2,
    ) -> None:
        super().__init__(
            relevance_threshold=relevance_threshold,
            use_rating_weights=use_rating_weights,
        )
        self.alpha = float(alpha)
        self.n_steps = int(n_steps)

    def predict(
        self,
        users: pd.DataFrame,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        k: int = 10,
    ) -> dict[int, list[Rating]]:
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
        positive_by_user = self._user_positive_sets(ratings)
        catalog_mask = self._catalog_mask(movies)
        trans_t = self._transition.T if self._transition is not None else None

        preds: Dict[int, List[Rating]] = {}
        for uid in users["UserID"].astype(int).values:
            seed = self._seed_vector(positive_by_user.get(int(uid), set()))
            score = seed.copy()
            if trans_t is not None and seed.size:
                for _ in range(max(0, self.n_steps)):
                    score = self.alpha * (trans_t @ score) + (1.0 - self.alpha) * seed

            item_ids = self.item_ids[catalog_mask]
            scores = score[catalog_mask] if score.size else np.array([], dtype=np.float64)
            preds[int(uid)] = self.top_k_from_scores(
                item_ids=item_ids,
                scores=scores,
                k=k,
                seen=seen_by_user.get(int(uid), set()),
            )

        return preds


class PageRankRanker(GraphRankerBase):
    """Global PageRank on item graph."""

    def __init__(
        self,
        relevance_threshold: float = 4.0,
        use_rating_weights: bool = False,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-8,
    ) -> None:
        super().__init__(
            relevance_threshold=relevance_threshold,
            use_rating_weights=use_rating_weights,
        )
        self.damping = float(damping)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self._pagerank: np.ndarray = np.array([], dtype=np.float64)

    def fit(
        self,
        ratings: pd.DataFrame,
        users: pd.DataFrame | None = None,
        movies: pd.DataFrame | None = None,
    ) -> "PageRankRanker":
        super().fit(ratings=ratings, users=users, movies=movies)
        n_items = len(self.item_ids)
        if n_items == 0:
            self._pagerank = np.array([], dtype=np.float64)
            return self

        trans_t = self._transition.T
        p = np.full(n_items, 1.0 / n_items, dtype=np.float64)
        teleport = np.full(n_items, 1.0 / n_items, dtype=np.float64)

        for _ in range(self.max_iter):
            new_p = self.damping * (trans_t @ p) + (1.0 - self.damping) * teleport
            if np.linalg.norm(new_p - p, ord=1) <= self.tol:
                p = new_p
                break
            p = new_p

        s = float(p.sum())
        self._pagerank = p / s if s > 0 else teleport
        return self

    def predict(
        self,
        users: pd.DataFrame,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        k: int = 10,
    ) -> dict[int, list[Rating]]:
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
        mask = self._catalog_mask(movies)
        item_ids = self.item_ids[mask]
        scores = self._pagerank[mask] if self._pagerank.size else np.array([], dtype=np.float64)

        preds: Dict[int, List[Rating]] = {}
        for uid in users["UserID"].astype(int).values:
            preds[int(uid)] = self.top_k_from_scores(
                item_ids=item_ids,
                scores=scores,
                k=k,
                seen=seen_by_user.get(int(uid), set()),
            )
        return preds


class PersonalizedPageRankRanker(GraphRankerBase):
    """User-personalized PageRank (random walk with restart)."""

    def __init__(
        self,
        relevance_threshold: float = 4.0,
        use_rating_weights: bool = False,
        damping: float = 0.85,
        max_iter: int = 50,
        tol: float = 1e-7,
    ) -> None:
        super().__init__(
            relevance_threshold=relevance_threshold,
            use_rating_weights=use_rating_weights,
        )
        self.damping = float(damping)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def predict(
        self,
        users: pd.DataFrame,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        k: int = 10,
    ) -> dict[int, list[Rating]]:
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
        positive_by_user = self._user_positive_sets(ratings)
        mask = self._catalog_mask(movies)
        trans_t = self._transition.T if self._transition is not None else None

        preds: Dict[int, List[Rating]] = {}
        for uid in users["UserID"].astype(int).values:
            seed = self._seed_vector(positive_by_user.get(int(uid), set()))
            if not seed.size:
                preds[int(uid)] = []
                continue

            p = seed.copy()
            if trans_t is not None:
                for _ in range(self.max_iter):
                    new_p = self.damping * (trans_t @ p) + (1.0 - self.damping) * seed
                    if np.linalg.norm(new_p - p, ord=1) <= self.tol:
                        p = new_p
                        break
                    p = new_p

            item_ids = self.item_ids[mask]
            scores = p[mask]
            preds[int(uid)] = self.top_k_from_scores(
                item_ids=item_ids,
                scores=scores,
                k=k,
                seen=seen_by_user.get(int(uid), set()),
            )

        return preds
