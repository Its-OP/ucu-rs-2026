from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd

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
    """Two-stage hybrid recommender: candidate union + LambdaMART reranking.

    Stage 1:
        - collaborative candidates from BPR (large pool)
        - content candidates from ContentBasedRecommender
        - union by item id

    Stage 2:
        - LightGBM LambdaMART reranker (lambdarank objective, optimises NDCG)
          over 16 CF/CB/popularity/user features with integer relevance grades.
    """

    _FEATURE_NAMES = [
        "cf_score_norm",           # min-max BPR score per user
        "cf_rank_inv",             # 1 / cf_rank
        "cf_rank_frac",            # cf_rank / n_cf_retrieved (relative position)
        "bpr_item_bias",           # raw BPR item bias (global popularity from CF)
        "cb_score_norm",           # min-max CB cosine similarity per user
        "cb_rank_inv",             # 1 / cb_rank
        "cb_rank_frac",            # cb_rank / n_cb_retrieved
        "in_cf",                   # binary: appears in CF candidates
        "in_cb",                   # binary: appears in CB candidates
        "cf_cb_interaction",       # cf_norm * cb_norm (agreement signal)
        "movie_mean_rating",       # global avg rating
        "log_movie_rating_count",  # log1p of rating count
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
        cf_candidates: int = 400,
        cb_candidates: int = 120,
        cb_search_size: int = 240,
        train_cf_candidates: int = 200,
        train_cb_candidates: int = 60,
        train_cb_search_size: int = 120,
        use_ranker: bool = True,
        blend_alpha: float = 0.7,
        ranker_cf_blend: float = 1.0,
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
        self.ranker_cf_blend = float(ranker_cf_blend)

        self.cf_model: BPRRecommender | None = None
        self.cb_model: ContentBasedRecommender | None = None
        self.ranker: lgb.LGBMRanker | None = None
        self._bpr_bias_lookup: Dict[int, float] = {}

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
        rerank_ratings: pd.DataFrame | None = None,
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
        # Build movie_id -> BPR item bias lookup for use as a reranker feature.
        if self.cf_model.item_bias is not None:
            self._bpr_bias_lookup = {
                int(self.cf_model.idx_to_item[idx]): float(self.cf_model.item_bias[idx])
                for idx in range(len(self.cf_model.item_bias))
            }

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
            train_labels = rerank_ratings if rerank_ratings is not None else ratings
            self._train_reranker(users=users, label_ratings=train_labels)

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

        user_ids = users["UserID"].astype(int).tolist()

        # Build all candidate signals first (per-user work).
        all_signals: Dict[int, list[CandidateSignals]] = {}
        for user_id in user_ids:
            signals = self._build_candidate_signals(
                user_id=user_id,
                n_cf=self.cf_candidates,
                n_cb=self.cb_candidates,
                cb_search_size=self.cb_search_size,
                filter_watched=True,
            )
            all_signals[user_id] = signals

        # Batch-score all users in a single ranker call to avoid per-user Python overhead.
        scored_map = self._score_all_candidates(all_signals)

        results: Dict[int, List[Rating]] = {}
        for user_id in user_ids:
            signals = all_signals[user_id]
            if not signals:
                results[user_id] = self._popular_fallback(user_id, k)
                continue

            scored = scored_map[user_id]
            scored.sort(key=lambda x: x[1], reverse=True)
            results[user_id] = [
                Rating(movie_id=int(movie_id), score=float(score))
                for movie_id, score in scored[:k]
            ]

        return results

    def _score_all_candidates(
        self,
        all_signals: Dict[int, list[CandidateSignals]],
    ) -> Dict[int, list[tuple[int, float]]]:
        """Score candidates for all users in a single batched ranker call."""
        users_with_signals = [uid for uid, sigs in all_signals.items() if sigs]

        if not users_with_signals:
            return {}

        # Fallback: ranker not trained, score per user via RRF (cheap).
        if self.ranker is None:
            return {
                uid: self._score_rank_fusion(all_signals[uid])
                for uid in users_with_signals
            }

        # Build one large feature matrix over all users, record block sizes.
        feature_blocks = [
            self._build_feature_matrix(uid, all_signals[uid])
            for uid in users_with_signals
        ]
        block_sizes = [b.shape[0] for b in feature_blocks]
        x = np.vstack(feature_blocks)

        # Single LGB predict call — much faster than 6k individual calls.
        scores = self.ranker.predict(x)

        def _minmax_1d(v: np.ndarray) -> np.ndarray:
            lo, hi = float(v.min()), float(v.max())
            if hi <= lo + 1e-12:
                return np.zeros_like(v, dtype=np.float64)
            return (v - lo) / (hi - lo)

        result: Dict[int, list[tuple[int, float]]] = {}
        offset = 0
        for uid, n in zip(users_with_signals, block_sizes):
            user_scores = scores[offset : offset + n]

            if self.ranker_cf_blend < 1.0:
                # Blend ranker score with the raw BPR CF score to preserve
                # BPR's strong top-position signal for NDCG@10 / MRR.
                cf_raw = np.array(
                    [all_signals[uid][i].cf_score for i in range(n)], dtype=np.float64
                )
                ranker_norm = _minmax_1d(user_scores.astype(np.float64))
                cf_norm = _minmax_1d(cf_raw)
                user_scores = (
                    self.ranker_cf_blend * ranker_norm
                    + (1.0 - self.ranker_cf_blend) * cf_norm
                )

            result[uid] = [
                (all_signals[uid][i].movie_id, float(user_scores[i]))
                for i in range(n)
            ]
            offset += n

        return result

    @staticmethod
    def _rating_to_grade(r: float) -> int:
        """Map a raw rating to a non-negative integer relevance grade for LambdaMART."""
        if r < 1.0:
            return 0  # unrated
        if r < 3.0:
            return 0  # 1-2 stars: irrelevant
        if r < 4.0:
            return 1  # 3 stars: somewhat relevant
        if r < 5.0:
            return 2  # 4 stars: relevant
        return 3      # 5 stars: highly relevant

    def _train_reranker(
        self,
        users: pd.DataFrame,
        label_ratings: pd.DataFrame,
    ) -> None:
        rating_lookup = {
            (int(uid), int(mid)): float(r)
            for uid, mid, r in zip(
                label_ratings["UserID"].values,
                label_ratings["MovieID"].values,
                label_ratings["Rating"].values,
            )
        }

        all_x: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        group_sizes: list[int] = []

        for user_id in users["UserID"].astype(int).tolist():
            signals = self._build_candidate_signals(
                user_id=user_id,
                n_cf=self.train_cf_candidates,
                n_cb=self.train_cb_candidates,
                cb_search_size=self.train_cb_search_size,
                filter_watched=True,
                seen_override=self.train_seen,
            )
            if not signals:
                continue

            features = self._build_feature_matrix(user_id, signals)
            grades = np.array(
                [
                    self._rating_to_grade(rating_lookup.get((user_id, s.movie_id), 0.0))
                    for s in signals
                ],
                dtype=np.int32,
            )

            # Skip groups with no positive candidates: LambdaMART can't learn from them.
            if np.all(grades == 0):
                continue

            all_x.append(features)
            all_y.append(grades)
            group_sizes.append(len(signals))

        if not all_x:
            logger.warning("No reranker samples collected; falling back to score blending.")
            self.ranker = None
            return

        x = np.vstack(all_x)
        y = np.concatenate(all_y)
        group = np.array(group_sizes, dtype=np.int32)
        pos_rate = float((y > 0).mean())

        self.ranker = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbose=-1,
        )
        self.ranker.fit(x, y, group=group)

        importance = dict(zip(self._FEATURE_NAMES, self.ranker.feature_importances_))
        logger.info(
            "Reranker trained: n_users=%d n_samples=%d n_features=%d positive_rate=%.4f",
            len(group_sizes),
            x.shape[0],
            x.shape[1],
            pos_rate,
        )
        logger.info("Reranker feature importances: %s", importance)

    def _build_candidate_signals(
        self,
        user_id: int,
        n_cf: int,
        n_cb: int,
        cb_search_size: int,
        filter_watched: bool,
        seen_override: dict[int, set[int]] | None = None,
    ) -> list[CandidateSignals]:
        cf_items = self._cf_candidates_for_user(
            user_id=user_id,
            k=n_cf,
            filter_watched=filter_watched,
            seen_override=seen_override,
        )
        cb_items = self._cb_candidates_for_user(
            user_id=user_id,
            n_candidates=n_cb,
            search_size=cb_search_size,
            filter_watched=filter_watched,
            seen_override=seen_override,
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
        seen_override: dict[int, set[int]] | None = None,
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
            seen_map = seen_override if seen_override is not None else self.train_seen
            watched = seen_map.get(user_id, set())
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
        seen_override: dict[int, set[int]] | None = None,
    ) -> list[tuple[int, float]]:
        if self.cb_model is None:
            return []
        if user_id not in self.cb_model.user_profiles:
            return []

        seen_map = seen_override if seen_override is not None else self.train_seen
        watched = seen_map.get(user_id, set())
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
            return self._score_rank_fusion(signals)

        x = self._build_feature_matrix(user_id, signals)
        scores = self.ranker.predict(x)
        return [
            (signals[i].movie_id, float(scores[i]))
            for i in range(len(signals))
        ]

    def _score_rank_fusion(
        self,
        signals: list[CandidateSignals],
    ) -> list[tuple[int, float]]:
        """Scale-safe fallback: weighted reciprocal rank fusion."""
        cf_w = max(0.0, min(1.0, self.blend_alpha))
        cb_w = 1.0 - cf_w
        # Larger constant smooths rank differences while preserving order.
        rrf_k = 60.0
        scored: list[tuple[int, float]] = []
        for s in signals:
            cf_term = cf_w / (rrf_k + s.cf_rank) if s.cf_rank > 0 else 0.0
            cb_term = cb_w / (rrf_k + s.cb_rank) if s.cb_rank > 0 else 0.0
            scored.append((s.movie_id, float(cf_term + cb_term)))
        return scored

    def _build_feature_matrix(
        self,
        user_id: int,
        signals: list[CandidateSignals],
    ) -> np.ndarray:
        user_mean, user_count, user_liked = self.user_stats.get(user_id, (0.0, 0, 0))
        is_cold = 1.0 if user_liked == 0 else 0.0
        x = np.zeros((len(signals), len(self._FEATURE_NAMES)), dtype=np.float32)

        cf_raw = np.array([s.cf_score for s in signals], dtype=np.float32)
        cb_raw = np.array([s.cb_score for s in signals], dtype=np.float32)

        def _minmax(values: np.ndarray) -> np.ndarray:
            vmin = float(values.min()) if values.size else 0.0
            vmax = float(values.max()) if values.size else 0.0
            if vmax <= vmin + 1e-12:
                return np.zeros_like(values, dtype=np.float32)
            return ((values - vmin) / (vmax - vmin)).astype(np.float32)

        cf_norm = _minmax(cf_raw)
        cb_norm = _minmax(cb_raw)

        # Denominator for relative rank fraction (avoid div-by-zero).
        n_cf = max(s.cf_rank for s in signals if s.in_cf) if any(s.in_cf for s in signals) else 1
        n_cb = max(s.cb_rank for s in signals if s.in_cb) if any(s.in_cb for s in signals) else 1

        for row, s in enumerate(signals):
            cf_rank_inv = 1.0 / float(s.cf_rank) if s.cf_rank > 0 else 0.0
            cb_rank_inv = 1.0 / float(s.cb_rank) if s.cb_rank > 0 else 0.0
            cf_rank_frac = float(s.cf_rank) / float(n_cf) if s.cf_rank > 0 else 1.0
            cb_rank_frac = float(s.cb_rank) / float(n_cb) if s.cb_rank > 0 else 1.0
            x[row, 0] = cf_norm[row]
            x[row, 1] = cf_rank_inv
            x[row, 2] = cf_rank_frac
            x[row, 3] = float(self._bpr_bias_lookup.get(s.movie_id, 0.0))
            x[row, 4] = cb_norm[row]
            x[row, 5] = cb_rank_inv
            x[row, 6] = cb_rank_frac
            x[row, 7] = float(s.in_cf)
            x[row, 8] = float(s.in_cb)
            x[row, 9] = float(cf_norm[row] * cb_norm[row])
            x[row, 10] = float(self.movie_avg.get(s.movie_id, 0.0))
            x[row, 11] = float(np.log1p(self.movie_rating_count.get(s.movie_id, 0)))
            x[row, 12] = float(user_mean)
            x[row, 13] = float(user_count)
            x[row, 14] = float(user_liked)
            x[row, 15] = float(is_cold)

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
