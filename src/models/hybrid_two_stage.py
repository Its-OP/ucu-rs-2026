from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.models.base import Rating, RecommenderModel
from src.models.bpr import BPRRecommender
from src.models.content_based import ContentBasedRecommender

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateSignals:
    movie_id: int
    cf_score: float = 0.0
    cf_rank: int = 0
    cb_score: float = 0.0
    cb_rank: int = 0
    in_cf: int = 0
    in_cb: int = 0


class TwoStageHybridRecommender(RecommenderModel):
    """Two-stage hybrid recommender: candidate union + GBR reranking.

    Stage 1:
        - collaborative candidates from BPR
        - content candidates from ContentBasedRecommender
        - union by item id

    Stage 2:
        - GradientBoostingRegressor over mixed CF/CB/popularity/user features
    """

    _FEATURE_NAMES = [
        "cf_score",
        "cf_rank_inv",
        "cb_score",
        "cb_rank_inv",
        "in_cf",
        "in_cb",
        "cf_cb_interaction",
        "movie_mean_rating",
        "movie_rating_count",
        "user_mean_rating",
        "user_rating_count",
        "user_liked_count",
        "is_cold_user",
    ]

    def __init__(
        self,
        threshold: float = 4.0,
        cf_n_factors: int = 64,
        cf_n_epochs: int = 20,
        cf_lr: float = 0.01,
        cf_regularization: float = 0.01,
        cf_negative_sampling: str = "uniform",
        cf_negative_pool: str = "unseen",
        cf_popularity_alpha: float = 0.75,
        random_state: int = 42,
        content_metric: str = "pearson",
        content_recency_decay: float = 1.3,
        content_n_neighbors: int = 12,
        content_min_liked: int = 5,
        content_min_ratings: int = 100,
        cf_candidates: int = 200,
        cb_candidates: int = 200,
        cb_search_size: int = 400,
        train_cf_candidates: int = 120,
        train_cb_candidates: int = 120,
        train_cb_search_size: int = 240,
        use_ranker: bool = True,
        blend_alpha: float = 0.7,
    ):
        self.threshold = float(threshold)
        self.cf_n_factors = int(cf_n_factors)
        self.cf_n_epochs = int(cf_n_epochs)
        self.cf_lr = float(cf_lr)
        self.cf_regularization = float(cf_regularization)
        self.cf_negative_sampling = cf_negative_sampling
        self.cf_negative_pool = cf_negative_pool
        self.cf_popularity_alpha = float(cf_popularity_alpha)
        self.random_state = int(random_state)

        self.content_metric = content_metric
        self.content_recency_decay = float(content_recency_decay)
        self.content_n_neighbors = int(content_n_neighbors)
        self.content_min_liked = int(content_min_liked)
        self.content_min_ratings = int(content_min_ratings)

        self.cf_candidates = int(cf_candidates)
        self.cb_candidates = int(cb_candidates)
        self.cb_search_size = int(cb_search_size)
        self.train_cf_candidates = int(train_cf_candidates)
        self.train_cb_candidates = int(train_cb_candidates)
        self.train_cb_search_size = int(train_cb_search_size)

        self.use_ranker = bool(use_ranker)
        self.blend_alpha = float(blend_alpha)

        self.cf_model: BPRRecommender | None = None
        self.cb_model: ContentBasedRecommender | None = None
        self.ranker: GradientBoostingRegressor | None = None

        self.movie_avg: Dict[int, float] = {}
        self.movie_rating_count: Dict[int, int] = {}
        self.user_stats: Dict[int, tuple[float, int, int]] = {}
        self.train_seen: Dict[int, set[int]] = {}
        self.global_popular: List[int] = []

    def fit(
        self,
        ratings: pd.DataFrame,
        users: pd.DataFrame | None = None,
        movies: pd.DataFrame | None = None,
    ) -> "TwoStageHybridRecommender":
        if users is None:
            raise ValueError("users dataframe is required")
        if movies is None:
            raise ValueError("movies dataframe is required")
        if "embedding" not in movies.columns:
            raise ValueError(
                "movies dataframe must contain 'embedding' for content-based component"
            )

        self._compute_stats(ratings)
        self.train_seen = {
            int(uid): set(g["MovieID"].astype(int).tolist())
            for uid, g in ratings.groupby("UserID")
        }

        logger.info("Fitting BPR component...")
        self.cf_model = BPRRecommender(
            n_factors=self.cf_n_factors,
            n_epochs=self.cf_n_epochs,
            lr=self.cf_lr,
            regularization=self.cf_regularization,
            threshold=self.threshold,
            negative_sampling=self.cf_negative_sampling,
            negative_pool=self.cf_negative_pool,
            popularity_alpha=self.cf_popularity_alpha,
            random_state=self.random_state,
        )
        self.cf_model.fit(ratings=ratings, users=users, movies=movies)

        logger.info("Fitting content-based component...")
        self.cb_model = ContentBasedRecommender(
            relevance_threshold=self.threshold,
            min_liked=self.content_min_liked,
            min_ratings=self.content_min_ratings,
            scoring="similarity",
            metric=self.content_metric,
            recency_decay=self.content_recency_decay,
            n_neighbors=self.content_n_neighbors,
        )
        self.cb_model.load(movies).fit(ratings)

        if self.use_ranker:
            logger.info("Training hybrid reranker...")
            self._train_reranker(users=users, ratings=ratings)

        return self

    def predict(
        self,
        users: pd.DataFrame,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        k: int = 10,
    ) -> Dict[int, List[Rating]]:
        if self.cf_model is None or self.cb_model is None:
            raise RuntimeError("Model not fitted. Call fit(...) first.")

        results: Dict[int, List[Rating]] = {}
        for user_id in users["UserID"].astype(int).tolist():
            signals = self._build_candidate_signals(
                user_id=user_id,
                n_cf=self.cf_candidates,
                n_cb=self.cb_candidates,
                cb_search_size=self.cb_search_size,
                filter_watched=True,
            )
            if not signals:
                results[user_id] = self._popular_fallback(user_id, k)
                continue

            scored = self._score_candidates(user_id, signals)
            scored.sort(key=lambda x: x[1], reverse=True)
            top = scored[:k]
            results[user_id] = [
                Rating(movie_id=int(movie_id), score=float(score))
                for movie_id, score in top
            ]

        return results

    def _train_reranker(self, users: pd.DataFrame, ratings: pd.DataFrame) -> None:
        rating_lookup = {
            (int(uid), int(mid)): float(r)
            for uid, mid, r in zip(
                ratings["UserID"].values,
                ratings["MovieID"].values,
                ratings["Rating"].values,
            )
        }

        all_x: list[np.ndarray] = []
        all_y: list[np.ndarray] = []

        for user_id in users["UserID"].astype(int).tolist():
            signals = self._build_candidate_signals(
                user_id=user_id,
                n_cf=self.train_cf_candidates,
                n_cb=self.train_cb_candidates,
                cb_search_size=self.train_cb_search_size,
                filter_watched=False,
            )
            if not signals:
                continue

            features = self._build_feature_matrix(user_id, signals)
            labels = np.array(
                [rating_lookup.get((user_id, s.movie_id), 0.0) for s in signals],
                dtype=np.float32,
            )

            if np.all(labels == 0.0):
                continue

            all_x.append(features)
            all_y.append(labels)

        if not all_x:
            logger.warning("No reranker samples collected; falling back to score blending.")
            self.ranker = None
            return

        x = np.vstack(all_x)
        y = np.concatenate(all_y)

        self.ranker = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=30,
            subsample=0.8,
            random_state=self.random_state,
        )
        self.ranker.fit(x, y)

        importance = dict(zip(self._FEATURE_NAMES, self.ranker.feature_importances_))
        logger.info(
            "Reranker trained: n_samples=%d n_features=%d",
            x.shape[0],
            x.shape[1],
        )
        logger.info("Reranker feature importances: %s", importance)

    def _build_candidate_signals(
        self,
        user_id: int,
        n_cf: int,
        n_cb: int,
        cb_search_size: int,
        filter_watched: bool,
    ) -> list[CandidateSignals]:
        cf_items = self._cf_candidates_for_user(
            user_id=user_id,
            k=n_cf,
            filter_watched=filter_watched,
        )
        cb_items = self._cb_candidates_for_user(
            user_id=user_id,
            n_candidates=n_cb,
            search_size=cb_search_size,
            filter_watched=filter_watched,
        )

        merged: dict[int, dict] = {}

        for rank, (movie_id, score) in enumerate(cf_items, start=1):
            current = merged.setdefault(
                int(movie_id),
                {
                    "cf_score": 0.0,
                    "cf_rank": 0,
                    "cb_score": 0.0,
                    "cb_rank": 0,
                    "in_cf": 0,
                    "in_cb": 0,
                },
            )
            current["cf_score"] = float(score)
            current["cf_rank"] = int(rank)
            current["in_cf"] = 1

        for rank, (movie_id, score) in enumerate(cb_items, start=1):
            current = merged.setdefault(
                int(movie_id),
                {
                    "cf_score": 0.0,
                    "cf_rank": 0,
                    "cb_score": 0.0,
                    "cb_rank": 0,
                    "in_cf": 0,
                    "in_cb": 0,
                },
            )
            current["cb_score"] = float(score)
            current["cb_rank"] = int(rank)
            current["in_cb"] = 1

        return [
            CandidateSignals(
                movie_id=int(mid),
                cf_score=float(v["cf_score"]),
                cf_rank=int(v["cf_rank"]),
                cb_score=float(v["cb_score"]),
                cb_rank=int(v["cb_rank"]),
                in_cf=int(v["in_cf"]),
                in_cb=int(v["in_cb"]),
            )
            for mid, v in merged.items()
        ]

    def _cf_candidates_for_user(
        self,
        user_id: int,
        k: int,
        filter_watched: bool,
    ) -> list[tuple[int, float]]:
        if self.cf_model is None:
            return []

        candidate_indices = self.cf_model._all_item_indices
        if candidate_indices.size == 0:
            return []

        item_ids = np.array(
            [int(self.cf_model.idx_to_item[i]) for i in candidate_indices], dtype=np.int64
        )
        item_bias = self.cf_model.item_bias[candidate_indices]
        item_factors = self.cf_model.item_factors[candidate_indices]

        if user_id in self.cf_model.user_to_idx:
            u_idx = self.cf_model.user_to_idx[user_id]
            scores = item_bias + (item_factors @ self.cf_model.user_factors[u_idx])
        else:
            scores = self.cf_model._global_popularity_scores[candidate_indices].astype(
                np.float32
            )

        work_scores = scores.astype(np.float64, copy=True)
        if filter_watched:
            watched = self.train_seen.get(user_id, set())
            if watched:
                mask = np.isin(item_ids, np.fromiter(watched, dtype=item_ids.dtype))
                work_scores[mask] = -np.inf

        valid_k = min(k, int(np.isfinite(work_scores).sum()))
        if valid_k <= 0:
            return []

        top_idx = np.argpartition(-work_scores, valid_k - 1)[:valid_k]
        top_idx = top_idx[np.argsort(-work_scores[top_idx])]

        return [
            (int(item_ids[i]), float(scores[i]))
            for i in top_idx
            if np.isfinite(work_scores[i])
        ]

    def _cb_candidates_for_user(
        self,
        user_id: int,
        n_candidates: int,
        search_size: int,
        filter_watched: bool,
    ) -> list[tuple[int, float]]:
        if self.cb_model is None:
            return []
        if user_id not in self.cb_model.user_profiles:
            return []

        watched = self.train_seen.get(user_id, set())
        profile = self.cb_model.user_profiles[user_id]
        candidates = self.cb_model._retrieve_candidates(
            profile=profile,
            search_size=max(search_size, n_candidates),
            n_candidates=n_candidates,
            filter_watched=filter_watched,
            watched=watched,
        )
        scored = self.cb_model._score_similarity(candidates)
        return [
            (int(self.cb_model.idx_to_movie_id[idx]), float(score))
            for idx, score in scored
        ]

    def _score_candidates(
        self,
        user_id: int,
        signals: list[CandidateSignals],
    ) -> list[tuple[int, float]]:
        if self.ranker is None:
            return [
                (
                    s.movie_id,
                    float(self.blend_alpha * s.cf_score + (1.0 - self.blend_alpha) * s.cb_score),
                )
                for s in signals
            ]

        x = self._build_feature_matrix(user_id, signals)
        scores = self.ranker.predict(x)
        return [
            (signals[i].movie_id, float(scores[i]))
            for i in range(len(signals))
        ]

    def _build_feature_matrix(
        self,
        user_id: int,
        signals: list[CandidateSignals],
    ) -> np.ndarray:
        user_mean, user_count, user_liked = self.user_stats.get(user_id, (0.0, 0, 0))
        is_cold = 1.0 if user_liked == 0 else 0.0
        x = np.zeros((len(signals), len(self._FEATURE_NAMES)), dtype=np.float32)

        for row, s in enumerate(signals):
            cf_rank_inv = 1.0 / float(s.cf_rank) if s.cf_rank > 0 else 0.0
            cb_rank_inv = 1.0 / float(s.cb_rank) if s.cb_rank > 0 else 0.0
            x[row, 0] = s.cf_score
            x[row, 1] = cf_rank_inv
            x[row, 2] = s.cb_score
            x[row, 3] = cb_rank_inv
            x[row, 4] = float(s.in_cf)
            x[row, 5] = float(s.in_cb)
            x[row, 6] = float(s.cf_score * s.cb_score)
            x[row, 7] = float(self.movie_avg.get(s.movie_id, 0.0))
            x[row, 8] = float(self.movie_rating_count.get(s.movie_id, 0))
            x[row, 9] = float(user_mean)
            x[row, 10] = float(user_count)
            x[row, 11] = float(user_liked)
            x[row, 12] = float(is_cold)

        return x

    def _compute_stats(self, ratings: pd.DataFrame) -> None:
        grouped_m = ratings.groupby("MovieID")["Rating"]
        self.movie_avg = grouped_m.mean().to_dict()
        self.movie_rating_count = grouped_m.count().to_dict()

        grouped_u = ratings.groupby("UserID")["Rating"]
        user_mean = grouped_u.mean()
        user_count = grouped_u.count()
        liked = (
            ratings.loc[ratings["Rating"] >= self.threshold]
            .groupby("UserID")
            .size()
        )
        self.user_stats = {
            int(uid): (
                float(user_mean[uid]),
                int(user_count[uid]),
                int(liked.get(uid, 0)),
            )
            for uid in user_mean.index
        }

        pop = (
            ratings.groupby("MovieID")["Rating"]
            .mean()
            .sort_values(ascending=False)
        )
        self.global_popular = [int(mid) for mid in pop.index.tolist()]

    def _popular_fallback(self, user_id: int, k: int) -> list[Rating]:
        watched = self.train_seen.get(user_id, set())
        recs = [mid for mid in self.global_popular if mid not in watched][:k]
        return [
            Rating(movie_id=int(mid), score=float(self.movie_avg.get(mid, 0.0)))
            for mid in recs
        ]
