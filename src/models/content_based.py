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
                 scoring: Literal["similarity", "mean_rating", "hybrid", "popular", "lambdamart"] = "similarity",
                 metric: Literal["cosine", "pearson"] = "cosine",
                 beta: float = 0.8,
                 recency_decay: float = 0.0,
                 n_neighbors: int = 0):
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
        self.global_top: list[int] = []  # movie_ids sorted by global avg rating
        self.global_avg: dict[int, float] = {}  # movie_id → global avg rating
        self.movie_avg: dict[int, float] = {}  # movie_id → mean rating (all movies)
        self.movie_rating_count: dict[int, int] = {}  # movie_id → rating count
        self.user_stats: dict[int, tuple[float, int, int]] = {}  # uid → (mean, count, liked_count)

        # Set by .train_ranker()
        self.ranker = None

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
        self.movie_rating_count = grouped.count().to_dict()

        # Per-user aggregate stats (for LambdaMART features)
        user_grouped = ratings.groupby("UserID")["Rating"]
        user_means = user_grouped.mean()
        user_counts = user_grouped.count()
        user_liked = (
            ratings[ratings["Rating"] > self.relevance_threshold]
            .groupby("UserID")
            .size()
        )
        self.user_stats = {
            uid: (float(user_means[uid]), int(user_counts[uid]), int(user_liked.get(uid, 0)))
            for uid in user_means.index
        }

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

        # Neighbor-based profile enrichment: average the user's profile
        # with their K nearest-neighbor profiles (equal weight for all,
        # including the user themselves).
        # This injects a lightweight collaborative filtering signal.
        if self.n_neighbors > 0 and len(self.user_profiles) > self.n_neighbors:
            user_ids = list(self.user_profiles.keys())
            profile_matrix = np.vstack(
                [self.user_profiles[uid] for uid in user_ids]
            ).astype("float32")                        # (n_users, dim)

            # Build a temporary FAISS index over user profiles
            dim = profile_matrix.shape[1]
            user_index = faiss.IndexFlatIP(dim)
            user_index.add(profile_matrix)

            # Query: K+1 neighbors (includes the user itself)
            sims, idxs = user_index.search(profile_matrix, self.n_neighbors + 1)

            for i, uid in enumerate(user_ids):
                # Average user + all K neighbors (self is included in results)
                group_profiles = profile_matrix[idxs[i]]  # (K+1, dim)
                enriched = group_profiles.mean(axis=0).reshape(1, -1).astype("float32")
                faiss.normalize_L2(enriched)
                self.user_profiles[uid] = enriched

        return self

    # -- LambdaMART feature names (positional in the ndarray) --
    _FEATURE_NAMES = [
        "cosine_similarity",
        "movie_mean_rating",
        "movie_rating_count",
        "user_mean_rating",
        "user_rating_count",
        "user_liked_count",
    ]

    def _compute_features(
        self,
        user_id: int,
        candidates: list[tuple[int, float]],
    ) -> np.ndarray:
        """
        Compute LambdaMART feature matrix for a list of (faiss_idx, cosine_sim)
        candidates belonging to one user.

        Returns ndarray of shape (n_candidates, 6).
        """
        n = len(candidates)
        features = np.zeros((n, 6), dtype=np.float32)

        u_mean, u_count, u_liked = self.user_stats.get(user_id, (0.0, 0, 0))

        for i, (idx, sim) in enumerate(candidates):
            movie_id = self.idx_to_movie_id[idx]
            features[i, 0] = sim
            features[i, 1] = self.movie_avg.get(movie_id, 0.0)
            features[i, 2] = self.movie_rating_count.get(movie_id, 0)
            features[i, 3] = u_mean
            features[i, 4] = u_count
            features[i, 5] = u_liked

        return features

    def train_ranker(
        self,
        ratings: pd.DataFrame,
        n_candidates: int = 300,
    ) -> "ContentBasedRecommender":
        """
        Train a LambdaMART re-ranker on the training data.

        Must be called after load() and fit().
        For each user with a profile, retrieves n_candidates from FAISS
        (WITHOUT filtering watched, so some candidates have known ratings),
        labels each with the user's actual rating (or 0), and trains
        an LGBMRanker with user-level groups.
        """
        from sklearn.ensemble import GradientBoostingRegressor
        from tqdm import tqdm

        if self.index is None or not self.user_profiles:
            raise RuntimeError("Call load() and fit() before train_ranker().")

        # Fast lookup: (user_id, movie_id) → rating
        rating_lookup: dict[tuple[int, int], float] = {}
        for uid, mid, r in zip(ratings["UserID"], ratings["MovieID"], ratings["Rating"]):
            rating_lookup[(int(uid), int(mid))] = float(r)

        all_features: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        search_size = n_candidates * 2

        for user_id, profile in tqdm(
            self.user_profiles.items(),
            desc="Building ranker training data",
            total=len(self.user_profiles),
        ):
            similarities, indices = self.index.search(profile, search_size)

            # Do NOT filter watched — we need rated movies as positive labels
            candidates = [
                (int(idx), float(sim))
                for idx, sim in zip(indices[0], similarities[0])
            ][:n_candidates]

            if len(candidates) == 0:
                continue

            features = self._compute_features(user_id, candidates)

            labels = np.zeros(len(candidates), dtype=np.float32)
            for i, (idx, _) in enumerate(candidates):
                movie_id = self.idx_to_movie_id[idx]
                labels[i] = rating_lookup.get((user_id, movie_id), 0.0)

            all_features.append(features)
            all_labels.append(labels)

        X_train = np.vstack(all_features)
        y_train = np.concatenate(all_labels)

        logger.info(
            "Reranker training: %d samples, %d features",
            X_train.shape[0], X_train.shape[1],
        )

        self.ranker = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=50,
            subsample=0.8,
            verbose=1,
        )

        self.ranker.fit(X_train, y_train)

        importance = dict(zip(self._FEATURE_NAMES, self.ranker.feature_importances_))
        logger.info("Feature importances: %s", importance)

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
            elif self.scoring == "lambdamart":
                if self.ranker is None:
                    raise RuntimeError(
                        "Ranker not trained. Call .train_ranker() before "
                        ".predict() with scoring='lambdamart'."
                    )
                features = self._compute_features(user_id, candidates)
                lgbm_scores = self.ranker.predict(features)
                scored = [
                    (candidates[i][0], float(lgbm_scores[i]))
                    for i in range(len(candidates))
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
