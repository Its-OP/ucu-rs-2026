from __future__ import annotations

import logging
from typing import Dict, List, Literal

import numpy as np
import pandas as pd

from .base import RecommenderModel, Rating

logger = logging.getLogger(__name__)


class BPRRecommender(RecommenderModel):
    """Bayesian Personalized Ranking with SGD and explicit negative sampling.

    Objective (BPR-OPT):
        maximize sum ln(sigmoid(x_ui - x_uj)) - lambda * ||Theta||^2
    for sampled triplets (u, i, j), where i is positive and j is negative.
    """

    def __init__(
        self,
        n_factors: int = 64,
        n_epochs: int = 20,
        lr: float = 0.01,
        regularization: float = 0.01,
        n_samples_per_epoch: int | None = None,
        threshold: float = 4.0,
        negative_sampling: Literal["uniform", "popularity"] = "uniform",
        negative_pool: Literal["unseen", "non_positive"] = "unseen",
        popularity_alpha: float = 0.75,
        random_state: int = 42,
    ):
        self.n_factors = int(n_factors)
        self.n_epochs = int(n_epochs)
        self.lr = float(lr)
        self.regularization = float(regularization)
        self.n_samples_per_epoch = n_samples_per_epoch
        self.threshold = float(threshold)
        self.negative_sampling = negative_sampling
        self.negative_pool = negative_pool
        self.popularity_alpha = float(popularity_alpha)
        self.random_state = int(random_state)

        self.user_to_idx: Dict[int, int] = {}
        self.item_to_idx: Dict[int, int] = {}
        self.idx_to_item: Dict[int, int] = {}

        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None

        self._all_item_indices: np.ndarray = np.array([], dtype=np.int64)
        self._train_seen_items_by_user: Dict[int, set[int]] = {}
        self._user_seen_idx_set: Dict[int, set[int]] = {}
        self._user_positive_idx: Dict[int, np.ndarray] = {}
        self._user_positive_set: Dict[int, set[int]] = {}
        self._users_with_positives: np.ndarray = np.array([], dtype=np.int64)
        self._item_sampling_probs: np.ndarray | None = None
        self._global_popularity_scores: np.ndarray = np.array([], dtype=np.float64)

        self.loss_history_: List[float] = []

    def _build_mappings(
        self,
        ratings: pd.DataFrame,
        movies: pd.DataFrame | None = None,
    ) -> None:
        users_list = ratings["UserID"].drop_duplicates().astype(int).to_numpy()
        if movies is not None and not movies.empty:
            if "MovieID" in movies.columns:
                items_list = movies["MovieID"].drop_duplicates().astype(int).to_numpy()
            elif "movie_id" in movies.columns:
                items_list = movies["movie_id"].drop_duplicates().astype(int).to_numpy()
            else:
                items_list = ratings["MovieID"].drop_duplicates().astype(int).to_numpy()
        else:
            items_list = ratings["MovieID"].drop_duplicates().astype(int).to_numpy()

        self.user_to_idx = {u: i for i, u in enumerate(users_list.tolist())}
        self.item_to_idx = {m: i for i, m in enumerate(items_list.tolist())}
        self.idx_to_item = {i: m for m, i in self.item_to_idx.items()}
        self._all_item_indices = np.arange(len(items_list), dtype=np.int64)

    def _build_positive_sets(self, ratings: pd.DataFrame) -> None:
        self._train_seen_items_by_user = {
            int(uid): set(group["MovieID"].astype(int).tolist())
            for uid, group in ratings.groupby("UserID")
        }
        self._user_seen_idx_set = {}
        for uid, seen_mids in self._train_seen_items_by_user.items():
            if uid not in self.user_to_idx:
                continue
            u_idx = self.user_to_idx[uid]
            self._user_seen_idx_set[u_idx] = {
                self.item_to_idx[mid] for mid in seen_mids if mid in self.item_to_idx
            }

        positives = ratings[ratings["Rating"] >= self.threshold].copy()
        if positives.empty:
            logger.warning(
                "No interactions found with rating >= %.2f. Falling back to all interactions as positives.",
                self.threshold,
            )
            positives = ratings.copy()

        self._user_positive_idx = {}
        self._user_positive_set = {}
        for uid, group in positives.groupby("UserID"):
            uid = int(uid)
            if uid not in self.user_to_idx:
                continue
            pos_idx = [
                self.item_to_idx[mid]
                for mid in group["MovieID"].astype(int).tolist()
                if mid in self.item_to_idx
            ]
            if pos_idx:
                uniq = list(set(pos_idx))
                u_idx = self.user_to_idx[uid]
                self._user_positive_idx[u_idx] = np.array(
                    uniq,
                    dtype=np.int64,
                )
                self._user_positive_set[u_idx] = set(uniq)

        self._users_with_positives = np.array(
            list(self._user_positive_idx.keys()),
            dtype=np.int64,
        )

        # popularity priors from observed interactions for cold users / popularity sampling
        counts = np.zeros(len(self.item_to_idx), dtype=np.float64)
        for mid, c in ratings["MovieID"].value_counts().items():
            idx = self.item_to_idx.get(int(mid))
            if idx is not None:
                counts[idx] = float(c)
        self._global_popularity_scores = counts.copy()

        pop = np.power(np.maximum(counts, 1.0), self.popularity_alpha)
        s = pop.sum()
        self._item_sampling_probs = pop / s if s > 0 else None

    def _sample_negative_item_idx(
        self,
        rng: np.random.Generator,
        blocked_item_set: set[int],
    ) -> int:
        if self.negative_sampling == "popularity" and self._item_sampling_probs is not None:
            for _ in range(10):
                j = int(rng.choice(self._all_item_indices, p=self._item_sampling_probs))
                if j not in blocked_item_set:
                    return j

        while True:
            j = int(rng.integers(0, len(self._all_item_indices)))
            if j not in blocked_item_set:
                return j

    @staticmethod
    def _safe_sigmoid(x: float) -> float:
        x = np.clip(x, -35.0, 35.0)
        return float(1.0 / (1.0 + np.exp(-x)))

    def fit(
        self,
        ratings: pd.DataFrame,
        users: pd.DataFrame | None = None,
        movies: pd.DataFrame | None = None,
    ) -> "BPRRecommender":
        """Fit BPR with SGD over sampled (u, i, j) triplets."""
        if ratings.empty:
            raise ValueError("ratings is empty")

        self._build_mappings(ratings=ratings, movies=movies)
        self._build_positive_sets(ratings=ratings)

        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)
        if n_users == 0 or n_items == 0:
            raise ValueError("No users/items available after mapping")
        if len(self._users_with_positives) == 0:
            raise ValueError("No users with positive interactions available for BPR")

        rng = np.random.default_rng(self.random_state)
        self.user_factors = rng.normal(0.0, 0.1, size=(n_users, self.n_factors)).astype(
            np.float32
        )
        self.item_factors = rng.normal(0.0, 0.1, size=(n_items, self.n_factors)).astype(
            np.float32
        )
        self.item_bias = np.zeros(n_items, dtype=np.float32)
        self.loss_history_.clear()

        default_samples = max(len(ratings), len(self._users_with_positives) * 10)
        n_samples = int(self.n_samples_per_epoch or default_samples)

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for _ in range(n_samples):
                u = int(rng.choice(self._users_with_positives))
                pos_idx = self._user_positive_idx[u]
                i = int(pos_idx[rng.integers(0, len(pos_idx))])
                if self.negative_pool == "unseen":
                    neg_block_set = self._user_seen_idx_set.get(u, set())
                elif self.negative_pool == "non_positive":
                    neg_block_set = self._user_positive_set[u]
                else:
                    raise ValueError(f"Unknown negative_pool: {self.negative_pool}")

                if len(neg_block_set) >= len(self._all_item_indices):
                    continue
                j = self._sample_negative_item_idx(
                    rng=rng,
                    blocked_item_set=neg_block_set,
                )

                pu = self.user_factors[u]
                qi = self.item_factors[i]
                qj = self.item_factors[j]
                bi = self.item_bias[i]
                bj = self.item_bias[j]

                x_ui = bi + float(pu @ qi)
                x_uj = bj + float(pu @ qj)
                x_uij = x_ui - x_uj

                # d/dx [-log(sigmoid(x))] = sigmoid(-x)
                grad = 1.0 - self._safe_sigmoid(x_uij)

                pu_old = pu.copy()
                qi_old = qi.copy()
                qj_old = qj.copy()

                self.user_factors[u] += self.lr * (
                    grad * (qi_old - qj_old) - self.regularization * pu_old
                )
                self.item_factors[i] += self.lr * (
                    grad * pu_old - self.regularization * qi_old
                )
                self.item_factors[j] += self.lr * (
                    -grad * pu_old - self.regularization * qj_old
                )
                self.item_bias[i] += self.lr * (grad - self.regularization * bi)
                self.item_bias[j] += self.lr * (-grad - self.regularization * bj)

                pair_loss = -np.log(self._safe_sigmoid(x_uij) + 1e-12)
                reg_term = self.regularization * (
                    float(np.dot(pu_old, pu_old))
                    + float(np.dot(qi_old, qi_old))
                    + float(np.dot(qj_old, qj_old))
                    + bi * bi
                    + bj * bj
                )
                epoch_loss += pair_loss + reg_term

            avg_loss = epoch_loss / n_samples
            self.loss_history_.append(float(avg_loss))
            if (epoch + 1) % 2 == 0 or epoch == 0:
                logger.info(
                    "BPR epoch %d/%d, avg loss: %.6f",
                    epoch + 1,
                    self.n_epochs,
                    avg_loss,
                )

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
        if self.user_factors is None or self.item_factors is None or self.item_bias is None:
            raise RuntimeError("Model is not fitted. Call fit(...) first.")

        seen_by_user = {
            int(uid): set(group["MovieID"].astype(int).tolist())
            for uid, group in ratings.groupby("UserID")
        }

        if "MovieID" in movies.columns:
            catalog = set(movies["MovieID"].astype(int).tolist())
        elif "movie_id" in movies.columns:
            catalog = set(movies["movie_id"].astype(int).tolist())
        else:
            catalog = set(self.item_to_idx.keys())

        candidate_indices = np.array(
            [idx for mid, idx in self.item_to_idx.items() if mid in catalog],
            dtype=np.int64,
        )
        if candidate_indices.size == 0:
            return {int(uid): [] for uid in users["UserID"].astype(int).values}

        item_ids = np.array([int(self.idx_to_item[i]) for i in candidate_indices], dtype=np.int64)
        item_bias = self.item_bias[candidate_indices]
        item_factors = self.item_factors[candidate_indices]

        preds: Dict[int, List[Rating]] = {}
        for uid in users["UserID"].astype(int).values:
            uid = int(uid)
            seen = seen_by_user.get(uid, set())

            if uid in self.user_to_idx:
                u_idx = self.user_to_idx[uid]
                scores = item_bias + (item_factors @ self.user_factors[u_idx])
            else:
                # Cold user fallback: global popularity prior
                scores = self._global_popularity_scores[candidate_indices].astype(np.float32)

            work_scores = scores.astype(np.float64, copy=True)
            if seen:
                mask = np.isin(item_ids, np.fromiter(seen, dtype=item_ids.dtype))
                work_scores[mask] = -np.inf

            valid_k = min(k, int(np.isfinite(work_scores).sum()))
            if valid_k == 0:
                preds[uid] = []
                continue

            top_idx = np.argpartition(-work_scores, valid_k - 1)[:valid_k]
            top_idx = top_idx[np.argsort(-work_scores[top_idx])]

            preds[uid] = [
                Rating(movie_id=int(item_ids[i]), score=float(scores[i]))
                for i in top_idx
                if np.isfinite(work_scores[i])
            ]

        return preds
