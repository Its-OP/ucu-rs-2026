import logging
from typing import Literal

import numpy as np
import pandas as pd
import faiss
from .base import RecommenderModel, Rating

logger = logging.getLogger(__name__)


class ContentBasedRecommender(RecommenderModel):
    """
    Content-based recommender:
    1. ANN (FAISS) retrieves candidates using user profile embedding
    2. Re-ranks candidates using one of two scoring strategies:
       - "similarity": scale cosine similarity into [relevance_threshold, 5]
       - "mean_rating": use each candidate's global mean rating
    """

    def __init__(self,
                 relevance_threshold: float = 4,
                 min_liked: int = 5,
                 min_ratings: int = 100,
                 scoring: Literal["similarity", "mean_rating", "hybrid", "popular"] = "similarity",
                 metric: Literal["cosine", "pearson"] = "cosine",
                 beta: float = 0.8,
                 recency_decay: float = 0.0):
        self.relevance_threshold = relevance_threshold
        self.min_liked = min_liked
        self.min_ratings = min_ratings
        self.scoring = scoring
        self.metric = metric
        self.beta = beta
        self.recency_decay = recency_decay

        # Set by .load()
        self.index: faiss.IndexFlatIP | None = None
        self.embeddings: np.ndarray | None = None
        self.movie_id_to_idx: dict[int, int] = {}
        self.idx_to_movie_id: dict[int, int] = {}

        # Set by .fit()
        self.user_profiles: dict[int, np.ndarray] = {}
        self.user_watched: dict[int, set[int]] = {}
        self.global_top: list[int] = []  # movie_ids sorted by global avg rating
        self.global_avg: dict[int, float] = {}  # movie_id → global avg rating
        self.movie_avg: dict[int, float] = {}  # movie_id → mean rating (all movies)

    def load(self, movies: pd.DataFrame) -> "ContentBasedRecommender":
        """
        Build FAISS index from movie embeddings.
        """
        movie_ids = movies["movie_id"].values
        self.embeddings = np.stack(movies["embedding"].values).astype("float32")

        self.movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
        self.idx_to_movie_id = {idx: movie_id for movie_id, idx in self.movie_id_to_idx.items()}

        # Mean-center for Pearson correlation (cosine on centered vectors)
        if self.metric == "pearson":
            self.embeddings -= self.embeddings.mean(axis=0)

        faiss.normalize_L2(self.embeddings)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

        return self

    def fit(self, ratings: pd.DataFrame) -> "ContentBasedRecommender":
        """
        Build user profiles.

        Search profiles are built from liked movies only (rating > relevance_threshold).
        """
        if self.embeddings is None:
            raise RuntimeError("Embeddings are missing. Call .load(movies) before .fit().")

        self.user_profiles = {}
        self.user_watched = {}

        # Pre-compute per-movie mean ratings
        known = ratings[ratings["MovieID"].isin(self.movie_id_to_idx)]
        grouped = known.groupby("MovieID")["Rating"]
        self.movie_avg = grouped.mean().to_dict()

        # Global top movies (filtered by min_ratings) for cold-start fallback
        avg_ratings = (
            grouped.mean()[grouped.count() >= self.min_ratings]
            .sort_values(ascending=False)
        )
        self.global_top = avg_ratings.index.tolist()
        self.global_avg = avg_ratings.to_dict()

        for user_id, group in ratings.groupby("UserID"):
            liked_embs = []
            liked_weights = []
            watched = set()

            # Sort chronologically so we can assign recency positions
            sorted_group = group.sort_values("Timestamp")
            n_ratings = len(sorted_group)

            for rank, (_, row) in enumerate(sorted_group.iterrows()):
                movie_id = row["MovieID"]
                if movie_id in self.movie_id_to_idx:
                    watched.add(movie_id)
                    if row["Rating"] > self.relevance_threshold:
                        idx = self.movie_id_to_idx[movie_id]
                        liked_embs.append(self.embeddings[idx])
                        # Non-linear weight: floor at 0.3, then square so
                        # 5-star (w=1.0) contributes ~3.3× more than 4.1-star (w=0.3)
                        raw_w = max(0.3, row["Rating"] - self.relevance_threshold)
                        rating_w = raw_w ** 2

                        # Recency: exponential decay based on how far back
                        # this rating is from the user's most recent one.
                        # age_frac=0 for the newest, =1 for the oldest.
                        # decay=0 disables recency (all weights = 1).
                        if self.recency_decay > 0 and n_ratings > 1:
                            age_frac = 1.0 - rank / (n_ratings - 1)
                            recency_w = np.exp(-self.recency_decay * age_frac)
                        else:
                            recency_w = 1.0

                        liked_weights.append(rating_w * recency_w)

            self.user_watched[user_id] = watched

            # Search profile: rating-weighted mean of liked-movie embeddings.
            # Higher-rated movies pull the profile disproportionately more.
            # Embeddings are already centered (if pearson) and L2-normalized,
            # so the profile just needs re-normalization after averaging.
            if len(liked_embs) >= self.min_liked:
                liked_embs = np.array(liked_embs)
                weights = np.array(liked_weights, dtype="float32")
                profile = np.average(liked_embs, axis=0, weights=weights)
                profile = profile.reshape(1, -1).astype("float32")
                faiss.normalize_L2(profile)
                self.user_profiles[user_id] = profile

        return self

    def predict(
        self,
        users: pd.DataFrame,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        k: int = 10,
        n_candidates: int = 300
    ) -> dict[int, list[Rating]]:
        """
        Generate top-k recommendations for each user.
        """
        if self.index is None:
            raise RuntimeError("Index is missing. Call .load(movies) before .predict()")
        if not self.user_profiles:
            raise RuntimeError("Profiles are missing. Call .fit(ratings) before .predict()")

        results = {}
        search_size = n_candidates * 2

        for user_id in users["UserID"].unique():
            watched = self.user_watched.get(user_id, set())

            # Popular baseline or users without a liked-movie profile: global top
            if self.scoring == "popular" or user_id not in self.user_profiles:
                top_movies = [
                    movie_id for movie_id in self.global_top if movie_id not in watched
                ][:k]
                results[user_id] = [
                    Rating(movie_id=int(movie_id), score=self.global_avg[movie_id])
                    for movie_id in top_movies
                ]
                continue

            profile = self.user_profiles[user_id]

            # Stage 1: ANN candidate retrieval (IndexFlatIP, higher = more similar)
            similarities, indices = self.index.search(profile, search_size)

            # Filter watched movies
            candidates = [
                (idx, sim)
                for idx, sim in zip(indices[0], similarities[0])
                if self.idx_to_movie_id[idx] not in watched
            ][: n_candidates]

            if len(candidates) == 0:
                results[user_id] = []
                continue

            # Stage 2: Score candidates
            if self.scoring == "similarity":
                # Scale cosine similarity from [0, 1] → [relevance_threshold, 5]
                lo = self.relevance_threshold
                hi = 5.0
                scored = [
                    (idx, lo + (hi - lo) * max(0.0, sim))
                    for idx, sim in candidates
                ]
            elif self.scoring == "hybrid":
                # Blend personalised similarity with global mean rating:
                # score = β * scaled_sim + (1 - β) * mean_rating
                lo = self.relevance_threshold
                hi = 5.0
                scored = [
                    (idx,
                     self.beta * (lo + (hi - lo) * max(0.0, sim))
                     + (1.0 - self.beta) * self.movie_avg.get(self.idx_to_movie_id[idx], lo))
                    for idx, sim in candidates
                ]
            else:  # mean_rating
                scored = [
                    (idx, self.movie_avg.get(self.idx_to_movie_id[idx], 0.0))
                    for idx, _ in candidates
                ]

            # Sort descending, take top-k
            scored.sort(key=lambda x: x[1], reverse=True)
            top_k = scored[:k]

            results[user_id] = [
                Rating(
                    movie_id=int(self.idx_to_movie_id[idx]),
                    score=float(score),
                )
                for idx, score in top_k
            ]

        return results
