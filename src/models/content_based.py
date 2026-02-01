import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import faiss
from .base import RecommenderModel, Rating


class ContentBasedRecommender(RecommenderModel):
    """
    Two-stage content-based recommender:
    1. ANN (FAISS) retrieves candidates using user profile embedding
    2. Per-user Ridge regression re-ranks candidates
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

        # Set by .load()
        self.index: faiss.IndexFlatIP | None = None
        self.embeddings: np.ndarray | None = None
        self.movie_id_to_idx: dict[int, int] = {}
        self.idx_to_movie_id: dict[int, int] = {}

        # Set by .fit()
        self.user_profiles: dict[int, np.ndarray] = {}
        self.user_regressors: dict[int, Ridge] = {}
        self.user_watched: dict[int, set[int]] = {}

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
        """
        if self.embeddings is None:
            raise RuntimeError("Embeddings are missing. Call .load(movies) before .fit().")

        self.user_profiles = {}
        self.user_regressors = {}
        self.user_watched = {}

        for user_id, group in ratings.groupby("UserID"):
            user_embs, user_scores, watched = [], [], set()

            for _, row in group.iterrows():
                movie_id = row["MovieID"]
                if movie_id in self.movie_id_to_idx:
                    idx = self.movie_id_to_idx[movie_id]
                    user_embs.append(self.embeddings[idx])
                    user_scores.append(row["Rating"])
                    watched.add(movie_id)

            self.user_watched[user_id] = watched

            if len(user_embs) < 2:
                continue

            user_embs = np.array(user_embs)
            user_scores = np.array(user_scores)

            # Make lowest-scored items have score '1'
            weights = user_scores - user_scores.min() + 1
            # Make weights sum up to '1'
            weights = weights / weights.sum()
            profile = np.average(user_embs, axis=0, weights=weights)
            profile = profile.reshape(1, -1).astype("float32")
            faiss.normalize_L2(profile)
            self.user_profiles[user_id] = profile

            # Per-user regressor
            regressor = Ridge(alpha=self.alpha)
            regressor.fit(user_embs, user_scores)
            self.user_regressors[user_id] = regressor

        return self

    def predict(
        self,
        users: pd.DataFrame,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        k: int = 10,
        n_candidates: int = 100
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
            if user_id not in self.user_profiles:
                results[user_id] = []
                continue

            profile = self.user_profiles[user_id]
            regressor = self.user_regressors[user_id]
            watched = self.user_watched.get(user_id, set())

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