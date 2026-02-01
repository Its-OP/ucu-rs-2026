import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import faiss
from .base import RecommenderModel, Rating

logger = logging.getLogger(__name__)


class ContentBasedRecommender(RecommenderModel):
    """
    Two-stage content-based recommender:
    1. ANN (FAISS) retrieves candidates using user profile embedding
    2. Per-user Ridge regression re-ranks candidates
    """

    def __init__(self,
                 alpha: float = 1.0,
                 relevance_threshold: float = 4,
                 min_liked: int = 5,
                 min_ratings: int = 100):
        self.alpha = alpha
        self.relevance_threshold = relevance_threshold
        self.min_liked = min_liked
        self.min_ratings = min_ratings

        # Set by .load()
        self.index: faiss.IndexFlatIP | None = None
        self.embeddings: np.ndarray | None = None
        self.movie_id_to_idx: dict[int, int] = {}
        self.idx_to_movie_id: dict[int, int] = {}

        # Set by .fit()
        self.user_profiles: dict[int, np.ndarray] = {}
        self.user_regressors: dict[int, Ridge] = {}
        self.user_watched: dict[int, set[int]] = {}
        self.global_top: list[int] = []  # movie_ids sorted by global avg rating
        self.global_avg: dict[int, float] = {}  # movie_id → global avg rating

    def load(self, movies: pd.DataFrame) -> "ContentBasedRecommender":
        """
        Build FAISS index from movie embeddings.
        """
        movie_ids = movies["movie_id"].values
        self.embeddings = np.stack(movies["embedding"].values).astype("float32")

        self.movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
        self.idx_to_movie_id = {idx: movie_id for movie_id, idx in self.movie_id_to_idx.items()}

        # Normalize for cosine similarity
        faiss.normalize_L2(self.embeddings)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

        return self

    def fit(self, ratings: pd.DataFrame) -> "ContentBasedRecommender":
        """
        Build user profiles and fit per-user regressors.

        Search profiles are built from liked movies only (rating > relevance_threshold).
        Ridge regressors are trained on the entire user history to avoid
        degenerating into predicting high scores only.
        """
        if self.embeddings is None:
            raise RuntimeError("Embeddings are missing. Call .load(movies) before .fit().")

        self.user_profiles = {}
        self.user_regressors = {}
        self.user_watched = {}

        # Pre-compute global top movies (by average rating) for cold-start fallback
        known = ratings[ratings["MovieID"].isin(self.movie_id_to_idx)]
        grouped = known.groupby("MovieID")["Rating"]
        avg_ratings = (
            grouped.mean()[grouped.count() >= self.min_ratings]
            .sort_values(ascending=False)
        )
        self.global_top = avg_ratings.index.tolist()
        self.global_avg = avg_ratings.to_dict()

        for user_id, group in ratings.groupby("UserID"):
            all_embs, all_scores, watched = [], [], set()
            liked_embs = []

            for _, row in group.iterrows():
                movie_id = row["MovieID"]
                if movie_id in self.movie_id_to_idx:
                    idx = self.movie_id_to_idx[movie_id]
                    emb = self.embeddings[idx]
                    score = row["Rating"]

                    all_embs.append(emb)
                    all_scores.append(score)
                    watched.add(movie_id)

                    if score > self.relevance_threshold:
                        liked_embs.append(emb)

            self.user_watched[user_id] = watched

            if len(all_embs) < 2:
                continue

            all_embs = np.array(all_embs)
            all_scores = np.array(all_scores)

            # Search profile: built from liked movies only
            if len(liked_embs) >= self.min_liked:
                liked_embs = np.array(liked_embs)
                profile = liked_embs.mean(axis=0)
                profile = profile.reshape(1, -1).astype("float32")
                faiss.normalize_L2(profile)
                self.user_profiles[user_id] = profile

            # Per-user regressor: trained on full history
            regressor = Ridge(alpha=self.alpha)
            regressor.fit(all_embs, all_scores)
            self.user_regressors[user_id] = regressor

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

            # Users without a liked-movie profile: fall back to global top
            if user_id not in self.user_profiles:
                top_movies = [
                    movie_id for movie_id in self.global_top if movie_id not in watched
                ][:k]
                results[user_id] = [
                    Rating(movie_id=int(movie_id), score=self.global_avg[movie_id])
                    for movie_id in top_movies
                ]
                continue

            profile = self.user_profiles[user_id]
            regressor = self.user_regressors[user_id]

            # Stage 1: ANN candidate retrieval
            _, indices = self.index.search(profile, search_size)

            # Filter watched movies
            candidate_indices = [
                idx for idx in indices[0]
                if self.idx_to_movie_id[idx] not in watched
            ][: n_candidates]

            if len(candidate_indices) == 0:
                results[user_id] = []
                continue

            # Stage 2: Re-rank with per-user regressor
            candidate_embs = self.embeddings[candidate_indices]
            predicted_scores = regressor.predict(candidate_embs)

            # Sort descending, take top-k
            top_k_order = np.argsort(predicted_scores)[::-1][:k]

            recommendations = [
                Rating(
                    movie_id=int(self.idx_to_movie_id[candidate_indices[i]]),
                    score=float(predicted_scores[i]),
                )
                for i in top_k_order
            ]

            results[user_id] = recommendations

        return results