import logging
from typing import Literal

import numpy as np
import pandas as pd
import faiss
from .base import RecommenderModel, Rating

from sklearn.ensemble import GradientBoostingRegressor

logger = logging.getLogger(__name__)


class ContentBasedRecommender(RecommenderModel):
    """
    Content-based recommender with optional GBR re-ranking.

    Pipeline:
        1. load()          – build FAISS index from movie embeddings
        2. fit()           – build user profiles from training ratings
        3. train_ranker()  – (optional) train a pointwise GBR re-ranker
        4. predict()       – retrieve candidates via ANN, score, return top-k
    """

    _FEATURE_NAMES = [
        "cosine_similarity",
        "movie_mean_rating",
        "movie_rating_count",
        "user_mean_rating",
        "user_rating_count",
        "user_liked_count",
    ]

    def __init__(
        self,
        relevance_threshold: float = 4,
        min_liked: int = 5,
        min_ratings: int = 100,
        scoring: Literal[
            "similarity", "mean_rating", "hybrid", "popular", "gbr_reranker"
        ] = "similarity",
        metric: Literal["cosine", "pearson"] = "cosine",
        beta: float = 0.9,
        recency_decay: float = 1.3,
        n_neighbors: int = 12,
    ):
        self.relevance_threshold = relevance_threshold
        self.min_liked = min_liked
        self.min_ratings = min_ratings
        self.scoring = scoring
        self.metric = metric
        self.beta = beta
        self.recency_decay = recency_decay
        self.n_neighbors = n_neighbors

        # Set by .load()
        self.index: faiss.IndexFlatIP | None = None
        self.embeddings: np.ndarray | None = None
        self.movie_id_to_idx: dict[int, int] = {}
        self.idx_to_movie_id: dict[int, int] = {}

        # Set by .fit()
        self.user_profiles: dict[int, np.ndarray] = {}
        self.user_watched: dict[int, set[int]] = {}
        self.global_top: list[int] = []
        self.global_avg: dict[int, float] = {}
        self.movie_avg: dict[int, float] = {}
        self.movie_rating_count: dict[int, int] = {}
        self.user_stats: dict[int, tuple[float, int, int]] = {}

        # Set by .train_ranker()
        self.ranker = None


    def load(self, movies: pd.DataFrame) -> "ContentBasedRecommender":
        """Build FAISS inner-product index from movie embeddings."""
        movie_ids = movies["movie_id"].values
        self.embeddings = np.stack(movies["embedding"].values).astype("float32")

        self.movie_id_to_idx = {
            movie_id: index for index, movie_id in enumerate(movie_ids)
        }
        self.idx_to_movie_id = {
            index: movie_id for movie_id, index in self.movie_id_to_idx.items()
        }

        if self.metric == "pearson":
            self.embeddings -= self.embeddings.mean(axis=0)

        faiss.normalize_L2(self.embeddings)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

        return self


    def fit(self, ratings: pd.DataFrame) -> "ContentBasedRecommender":
        """Build per-user search profiles from training ratings."""
        if self.embeddings is None:
            raise RuntimeError("Call .load(movies) before .fit().")

        self.user_profiles = {}
        self.user_watched = {}

        self._compute_movie_stats(ratings)
        self._compute_user_stats(ratings)
        self._compute_global_top(ratings)

        for user_id, group in ratings.groupby("UserID"):
            profile, watched = self._build_user_profile(user_id, group)
            self.user_watched[user_id] = watched
            if profile is not None:
                self.user_profiles[user_id] = profile

        if self.n_neighbors > 0:
            self._enrich_profiles_with_neighbors()

        return self


    def _compute_movie_stats(self, ratings: pd.DataFrame) -> None:
        """Compute per-movie mean rating and rating count."""
        known = ratings[ratings["MovieID"].isin(self.movie_id_to_idx)]
        grouped = known.groupby("MovieID")["Rating"]
        self.movie_avg = grouped.mean().to_dict()
        self.movie_rating_count = grouped.count().to_dict()


    def _compute_user_stats(self, ratings: pd.DataFrame) -> None:
        """Compute per-user aggregate stats: (mean_rating, count, liked_count)."""
        user_grouped = ratings.groupby("UserID")["Rating"]
        user_means = user_grouped.mean()
        user_counts = user_grouped.count()
        user_liked = (
            ratings[ratings["Rating"] > self.relevance_threshold]
            .groupby("UserID")
            .size()
        )
        self.user_stats = {
            user_id: (
                float(user_means[user_id]),
                int(user_counts[user_id]),
                int(user_liked.get(user_id, 0)),
            )
            for user_id in user_means.index
        }


    def _compute_global_top(self, ratings: pd.DataFrame) -> None:
        """Compute global top movies (filtered by min_ratings) for cold-start."""
        known = ratings[ratings["MovieID"].isin(self.movie_id_to_idx)]
        grouped = known.groupby("MovieID")["Rating"]
        avg_ratings = (
            grouped.mean()[grouped.count() >= self.min_ratings]
            .sort_values(ascending=False)
        )
        self.global_top = avg_ratings.index.tolist()
        self.global_avg = avg_ratings.to_dict()


    def _build_user_profile(
        self,
        user_id: int,
        group: pd.DataFrame,
    ) -> tuple[np.ndarray | None, set[int]]:
        """
        Build a single user's search profile from their ratings.

        Returns (profile_vector_or_None, watched_movie_ids).
        """
        liked_embs: list[np.ndarray] = []
        liked_weights: list[float] = []
        watched: set[int] = set()

        sorted_group = group.sort_values("Timestamp")
        n_ratings = len(sorted_group)

        for rank, (_, row) in enumerate(sorted_group.iterrows()):
            movie_id = row["MovieID"]
            if movie_id not in self.movie_id_to_idx:
                continue

            watched.add(movie_id)

            if row["Rating"] <= self.relevance_threshold:
                continue

            index = self.movie_id_to_idx[movie_id]
            liked_embs.append(self.embeddings[index])

            weight = self._rating_weight(
                rating=row["Rating"],
                rank=rank,
                n_ratings=n_ratings,
            )
            liked_weights.append(weight)

        if len(liked_embs) < self.min_liked:
            return None, watched

        emb_matrix = np.array(liked_embs)
        weights = np.array(liked_weights, dtype="float32")
        profile = np.average(emb_matrix, axis=0, weights=weights)
        profile = profile.reshape(1, -1).astype("float32")
        faiss.normalize_L2(profile)

        return profile, watched


    def _rating_weight(self, rating: float, rank: int, n_ratings: int) -> float:
        """
        Compute the combined rating x recency weight for a single liked movie.

        Rating weight: floor at 0.3, then square -- so 5-star (w=1.0) contributes
        ~3.3x more than 4.1-star (w=0.3).

        Recency weight: exponential decay based on chronological position.
        age_frac=0 for the newest, =1 for the oldest.
        """
        raw_weight = max(0.3, rating - self.relevance_threshold)
        rating_weight = raw_weight ** 2

        if self.recency_decay > 0 and n_ratings > 1:
            age_frac = 1.0 - rank / (n_ratings - 1)
            recency_weight = np.exp(-self.recency_decay * age_frac)
        else:
            recency_weight = 1.0

        return rating_weight * recency_weight


    def _enrich_profiles_with_neighbors(self) -> None:
        """
        Average each user's profile with their K nearest-neighbor profiles.
        """
        if len(self.user_profiles) <= self.n_neighbors:
            return

        user_ids = list(self.user_profiles.keys())
        profile_matrix = np.vstack(
            [self.user_profiles[user_id] for user_id in user_ids]
        ).astype("float32")

        dim = profile_matrix.shape[1]
        user_index = faiss.IndexFlatIP(dim)
        user_index.add(profile_matrix)

        # K+1 because the user itself appears in results
        _sims, neighbor_indices = user_index.search(
            profile_matrix, self.n_neighbors + 1
        )

        for position, user_id in enumerate(user_ids):
            neighbor_profiles = profile_matrix[neighbor_indices[position]]  # (K+1, dim)
            enriched = neighbor_profiles.mean(axis=0).reshape(1, -1).astype("float32")
            faiss.normalize_L2(enriched)
            self.user_profiles[user_id] = enriched


    def train_ranker(
        self,
        ratings: pd.DataFrame,
        n_candidates: int = 300,
    ) -> "ContentBasedRecommender":
        """
        Train a pointwise GradientBoostingRegressor re-ranker.

        For each user, retrieves n_candidates from FAISS *without* filtering
        watched movies (so rated movies appear as positive labels).
        Unrated candidates are labelled 0.
        """

        if self.index is None or not self.user_profiles:
            raise RuntimeError("Call load() and fit() before train_ranker().")

        rating_lookup = self._build_rating_lookup(ratings)

        all_features: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        search_size = n_candidates * 2

        for user_id, profile in self.user_profiles.items():
            candidates = self._retrieve_candidates(
                profile, search_size, n_candidates, filter_watched=False,
            )
            if not candidates:
                continue

            features = self._compute_features(user_id, candidates)
            labels = self._label_candidates(user_id, candidates, rating_lookup)

            all_features.append(features)
            all_labels.append(labels)

        feature_matrix = np.vstack(all_features)
        label_vector = np.concatenate(all_labels)

        logger.info(
            "Reranker training: %d samples, %d features",
            feature_matrix.shape[0],
            feature_matrix.shape[1],
        )

        self.ranker = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=50,
            subsample=0.8,
            verbose=1,
        )
        self.ranker.fit(feature_matrix, label_vector)

        importance = dict(zip(self._FEATURE_NAMES, self.ranker.feature_importances_))
        logger.info("Feature importances: %s", importance)

        return self


    @staticmethod
    def _build_rating_lookup(
        ratings: pd.DataFrame,
    ) -> dict[tuple[int, int], float]:
        """Build a fast (user_id, movie_id) -> rating lookup dict."""
        return {
            (int(user_id), int(movie_id)): float(rating)
            for user_id, movie_id, rating in zip(
                ratings["UserID"], ratings["MovieID"], ratings["Rating"]
            )
        }


    def _retrieve_candidates(
        self,
        profile: np.ndarray,
        search_size: int,
        n_candidates: int,
        filter_watched: bool = True,
        watched: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """
        Retrieve candidate movies from FAISS for a single user profile.

        Returns a list of (faiss_idx, cosine_similarity) tuples.
        """
        similarities, indices = self.index.search(profile, search_size)

        if filter_watched and watched:
            candidates = [
                (int(index), float(similarity))
                for index, similarity in zip(indices[0], similarities[0])
                if self.idx_to_movie_id[index] not in watched
            ][:n_candidates]
        else:
            candidates = [
                (int(index), float(similarity))
                for index, similarity in zip(indices[0], similarities[0])
            ][:n_candidates]

        return candidates


    def _compute_features(
        self,
        user_id: int,
        candidates: list[tuple[int, float]],
    ) -> np.ndarray:
        """
        Compute feature matrix for a list of (faiss_idx, cosine_sim) candidates.

        Returns ndarray of shape (n_candidates, len(_FEATURE_NAMES)).
        """
        n_candidates_actual = len(candidates)
        features = np.zeros(
            (n_candidates_actual, len(self._FEATURE_NAMES)), dtype=np.float32
        )

        user_mean, user_count, user_liked = self.user_stats.get(
            user_id, (0.0, 0, 0)
        )

        for position, (index, similarity) in enumerate(candidates):
            movie_id = self.idx_to_movie_id[index]
            features[position, 0] = similarity
            features[position, 1] = self.movie_avg.get(movie_id, 0.0)
            features[position, 2] = self.movie_rating_count.get(movie_id, 0)
            features[position, 3] = user_mean
            features[position, 4] = user_count
            features[position, 5] = user_liked

        return features


    def _label_candidates(
        self,
        user_id: int,
        candidates: list[tuple[int, float]],
        rating_lookup: dict[tuple[int, int], float],
    ) -> np.ndarray:
        """Label each candidate with the user's actual rating (or 0 if unrated)."""
        labels = np.zeros(len(candidates), dtype=np.float32)
        for position, (index, _) in enumerate(candidates):
            movie_id = self.idx_to_movie_id[index]
            labels[position] = rating_lookup.get((user_id, movie_id), 0.0)
        return labels


    def predict(
        self,
        users: pd.DataFrame,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        k: int = 10,
        n_candidates: int = 300,
    ) -> dict[int, list[Rating]]:
        """Generate top-k recommendations for each user."""
        if self.index is None:
            raise RuntimeError("Call .load(movies) before .predict().")
        if not self.user_profiles:
            raise RuntimeError("Call .fit(ratings) before .predict().")

        results: dict[int, list[Rating]] = {}
        search_size = n_candidates * 2

        for user_id in users["UserID"].unique():
            watched = self.user_watched.get(user_id, set())

            if self.scoring == "popular" or user_id not in self.user_profiles:
                results[user_id] = self._popular_fallback(watched, k)
                continue

            profile = self.user_profiles[user_id]
            candidates = self._retrieve_candidates(
                profile, search_size, n_candidates,
                filter_watched=True, watched=watched,
            )

            if not candidates:
                results[user_id] = []
                continue

            scored = self._score_candidates(user_id, candidates)

            scored.sort(key=lambda x: x[1], reverse=True)
            top_k = scored[:k]

            results[user_id] = [
                Rating(
                    movie_id=int(self.idx_to_movie_id[index]),
                    score=float(score),
                )
                for index, score in top_k
            ]

        return results


    def _popular_fallback(self, watched: set[int], k: int) -> list[Rating]:
        """Return top-k globally popular movies the user hasn't watched."""
        top_movies = [
            movie_id for movie_id in self.global_top if movie_id not in watched
        ][:k]
        return [
            Rating(movie_id=int(movie_id), score=self.global_avg[movie_id])
            for movie_id in top_movies
        ]


    def _score_candidates(
        self,
        user_id: int,
        candidates: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        """
        Score a list of (faiss_idx, cosine_sim) candidates using the
        configured scoring strategy.
        """
        if self.scoring == "similarity":
            return self._score_similarity(candidates)
        elif self.scoring == "hybrid":
            return self._score_hybrid(candidates)
        elif self.scoring == "gbr_reranker":
            return self._score_gbr_reranker(user_id, candidates)
        else:
            return self._score_mean_rating(candidates)


    def _score_similarity(
        self, candidates: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        """Scale cosine similarity from [0, 1] -> [relevance_threshold, 5]."""
        low, high = self.relevance_threshold, 5.0
        return [
            (index, low + (high - low) * max(0.0, similarity))
            for index, similarity in candidates
        ]


    def _score_hybrid(
        self, candidates: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        """Blend personalised similarity with global mean rating."""
        low, high = self.relevance_threshold, 5.0
        return [
            (
                index,
                self.beta * (low + (high - low) * max(0.0, similarity))
                + (1.0 - self.beta)
                * self.movie_avg.get(self.idx_to_movie_id[index], low),
            )
            for index, similarity in candidates
        ]


    def _score_gbr_reranker(
        self,
        user_id: int,
        candidates: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        """Score candidates with the trained GBR re-ranker."""
        if self.ranker is None:
            raise RuntimeError(
                "Ranker not trained. Call .train_ranker() before "
                ".predict() with scoring='gbr_reranker'."
            )
        features = self._compute_features(user_id, candidates)
        scores = self.ranker.predict(features)
        return [
            (candidates[position][0], float(scores[position]))
            for position in range(len(candidates))
        ]


    def _score_mean_rating(
        self, candidates: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        """Score each candidate by its global mean rating."""
        return [
            (index, self.movie_avg.get(self.idx_to_movie_id[index], 0.0))
            for index, _ in candidates
        ]
