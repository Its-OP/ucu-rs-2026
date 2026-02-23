"""Two-tower transformer recommender model.

Combines a lightweight transformer-based user tower with an MLP-based item
tower for top-K recommendation.  Both towers project raw 1536-dim SentenceBERT
concatenated embeddings into a shared 128-dim scoring space via independent
learned projections.  Scoring is dot product (equivalent to cosine similarity
since both outputs are L2-normalised).

FAISS is used **only at inference time** for efficient approximate nearest
neighbour retrieval; during training, dot products are computed directly so
that gradients flow through both tower projections.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List

# Prevent OpenMP conflict between torch and faiss-cpu on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# FAISS is imported lazily (inside methods that use it) to avoid an OpenMP
# conflict on macOS: both torch and faiss-cpu ship their own libomp.dylib and
# importing faiss *before* torch initialises OpenMP causes a segfault.
# By deferring the faiss import until after torch is fully loaded we sidestep
# the issue.  The import is cached by Python so it's only resolved once.

from src.models.base import Rating, RecommenderModel
from src.models.ANN.two_towers.item_tower import ItemTower
from src.models.ANN.two_towers.user_tower import (
    AGE_BUCKET_TO_INDEX,
    GENDER_TO_INDEX,
    UserTower,
)

logger = logging.getLogger(__name__)


def _detect_device(preference: str = "auto") -> torch.device:
    """Select the best available compute device.

    Parameters
    ----------
    preference : str
        One of ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.

    Returns
    -------
    torch.device
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


class TwoTowerTransformerRecommender(RecommenderModel):
    """Two-tower recommender with a transformer user encoder.

    Call ``prepare()`` before training or inference to initialise internal
    state (embeddings, mappings, towers, device placement).

    The ``fit()`` method is intentionally **not** defined here.  The training
    loop lives in the training script (``src/train_two_tower_transformer.py``)
    to keep the model class focused on architecture and inference.

    After training, call ``predict()`` (the ``RecommenderModel`` ABC method)
    to generate top-K recommendations backed by a FAISS index.
    """

    def __init__(
        self,
        # ── Architecture ─────────────────────────────────────────────
        projection_dimension: int = 128,
        item_hidden_dimension: int = 256,
        number_of_transformer_layers: int = 2,
        number_of_attention_heads: int = 4,
        feedforward_dimension: int = 256,
        maximum_history_length: int = 64,
        dropout_rate: float = 0.1,
        demographics_embedding_dimension: int = 16,
        rating_embedding_dimension: int = 16,
        item_embedding_dimension: int = 1536,
        # ── Inference ────────────────────────────────────────────────
        positive_threshold: float = 4.0,
        faiss_search_candidates: int = 300,
        inference_batch_size: int = 128,
        # ── Infrastructure ───────────────────────────────────────────
        device: str = "auto",
    ) -> None:
        self.projection_dimension = projection_dimension
        self.item_embedding_dimension = item_embedding_dimension
        self.maximum_history_length = maximum_history_length
        self.positive_threshold = positive_threshold
        self.faiss_search_candidates = faiss_search_candidates
        self.inference_batch_size = inference_batch_size

        self.device = _detect_device(device)

        # ── Towers ───────────────────────────────────────────────────
        self.item_tower = ItemTower(
            input_dimension=item_embedding_dimension,
            projection_dimension=projection_dimension,
            hidden_dimension=item_hidden_dimension,
            dropout_rate=dropout_rate,
        )
        self.user_tower = UserTower(
            item_embedding_dimension=item_embedding_dimension,
            projection_dimension=projection_dimension,
            number_of_transformer_layers=number_of_transformer_layers,
            number_of_attention_heads=number_of_attention_heads,
            feedforward_dimension=feedforward_dimension,
            maximum_history_length=maximum_history_length,
            dropout_rate=dropout_rate,
            demographics_embedding_dimension=demographics_embedding_dimension,
            rating_embedding_dimension=rating_embedding_dimension,
        )

        # ── State populated by prepare() ─────────────────────────────
        self.raw_embeddings: np.ndarray | None = None  # (n_movies, 1536)
        self.movie_id_to_index: Dict[int, int] = {}
        self.index_to_movie_id: Dict[int, int] = {}

        # ── FAISS index (built on demand at inference) ───────────────
        self.faiss_index = None  # faiss.IndexFlatIP, lazily created

        # ── Popularity fallback for cold-start users ─────────────────
        self._popularity_scores: np.ndarray | None = None
        self._popularity_movie_ids: np.ndarray | None = None

    # ─────────────────────────────────────────────────────────────────────
    # Preparation
    # ─────────────────────────────────────────────────────────────────────

    def prepare(
        self,
        movies_enriched: pd.DataFrame,
        raw_embeddings: np.ndarray,
        train_ratings: pd.DataFrame | None = None,
    ) -> "TwoTowerTransformerRecommender":
        """Initialise embeddings, mappings, and move towers to device.

        Parameters
        ----------
        movies_enriched : pd.DataFrame
            Must contain a ``movie_id`` column whose ordering matches the rows
            of ``raw_embeddings``.
        raw_embeddings : np.ndarray
            Raw concatenated SentenceBERT embeddings, shape
            ``(number_of_movies, 1536)``.
        train_ratings : pd.DataFrame | None
            If provided, used to pre-compute popularity fallback scores.

        Returns
        -------
        TwoTowerTransformerRecommender
            ``self``, for chaining.
        """
        movie_ids = movies_enriched["movie_id"].values.astype(int)
        self.movie_id_to_index = {
            int(movie_id): index for index, movie_id in enumerate(movie_ids)
        }
        self.index_to_movie_id = {
            index: int(movie_id)
            for movie_id, index in self.movie_id_to_index.items()
        }

        self.raw_embeddings = raw_embeddings.astype(np.float32)

        if train_ratings is not None:
            self._build_popularity_fallback(train_ratings)

        self.item_tower.to(self.device)
        self.user_tower.to(self.device)

        logger.info(
            "Prepared TwoTowerTransformerRecommender: %d movies, device=%s",
            len(self.movie_id_to_index),
            self.device,
        )
        return self

    def _build_popularity_fallback(self, ratings: pd.DataFrame) -> None:
        """Pre-compute global mean-rating scores for cold-start users."""
        known = ratings[ratings["MovieID"].isin(self.movie_id_to_index)]
        grouped = known.groupby("MovieID")["Rating"]
        means = grouped.mean()
        counts = grouped.count()
        # Only consider movies with at least 10 ratings for the fallback.
        eligible = means[counts >= 10].sort_values(ascending=False)
        self._popularity_movie_ids = np.array(
            eligible.index.tolist(), dtype=np.int64,
        )
        self._popularity_scores = eligible.values.astype(np.float64)

    # ─────────────────────────────────────────────────────────────────────
    # FAISS index management
    # ─────────────────────────────────────────────────────────────────────

    def build_faiss_index(self) -> None:
        """(Re-)build the FAISS inner-product index from projected items.

        Uses the item tower in inference mode (no gradients).  This must be
        called before ``predict()`` and should be refreshed whenever the item
        tower weights change (e.g. after each evaluation checkpoint during
        training).
        """
        if self.raw_embeddings is None:
            raise RuntimeError("Call prepare() before build_faiss_index().")

        import faiss  # Lazy import — see module-level comment about OpenMP.

        # On macOS, torch and faiss-cpu each ship their own libomp.dylib.
        # Running FAISS with multi-threaded OMP after torch has initialised its
        # own OMP runtime causes a segfault.  Restricting FAISS to 1 OMP thread
        # avoids the crash while having negligible impact on the small
        # IndexFlatIP index (3,883 items).
        faiss.omp_set_num_threads(1)

        projected = self.item_tower.project_all_items(self.raw_embeddings)
        # projected is already L2-normalised by ItemTower.forward().
        self.faiss_index = faiss.IndexFlatIP(self.projection_dimension)
        self.faiss_index.add(projected)

        logger.info(
            "FAISS index built: %d items, dimension=%d",
            self.faiss_index.ntotal,
            self.projection_dimension,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Inference (RecommenderModel ABC)
    # ─────────────────────────────────────────────────────────────────────

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
            Mapping from UserID to a list of Rating objects sorted by score
            descending (length up to k).
        """
        if self.faiss_index is None:
            self.build_faiss_index()

        # Build per-user seen-items sets for filtering (vectorised).
        t0 = time.time()
        seen_by_user: Dict[int, set[int]] = {}
        for user_id, movie_ids in ratings.groupby("UserID")["MovieID"]:
            seen_by_user[int(user_id)] = set(movie_ids.astype(int))
        logger.debug("predict: seen_by_user built in %.2fs", time.time() - t0)

        # Build per-user history data (all rated items, sorted most-recent-first).
        t0 = time.time()
        user_histories = self._build_user_histories(ratings)
        logger.debug(
            "predict: user_histories built in %.2fs (%d users)",
            time.time() - t0,
            len(user_histories),
        )

        # Pre-index users DataFrame once for efficient per-user lookups.
        users_indexed = users.set_index("UserID")

        # Encode users in batches and query FAISS.
        user_ids = users["UserID"].astype(int).values.tolist()
        predictions: Dict[int, List[Rating]] = {}

        t0 = time.time()
        for batch_start in range(0, len(user_ids), self.inference_batch_size):
            batch_user_ids = user_ids[
                batch_start : batch_start + self.inference_batch_size
            ]
            batch_predictions = self._predict_batch(
                batch_user_ids=batch_user_ids,
                user_histories=user_histories,
                users_dataframe=users_indexed,
                seen_by_user=seen_by_user,
                k=k,
            )
            predictions.update(batch_predictions)

        logger.debug(
            "predict: user encoding + FAISS retrieval in %.2fs (%d users)",
            time.time() - t0,
            len(predictions),
        )
        return predictions

    def _build_user_histories(
        self,
        ratings: pd.DataFrame,
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """Build per-user history arrays from interactions.

        Uses vectorised pandas/numpy operations instead of ``iterrows()``
        for dramatically better performance (~100x faster on large datasets).

        Returns a dict mapping user_id to a dict with keys:
        - ``"embeddings"`` : np.ndarray of shape ``(history_length, 1536)``
        - ``"rating_indices"`` : np.ndarray of shape ``(history_length,)``
        """
        # Pre-filter to movies in the catalogue and map to embedding indices.
        movie_id_series = ratings["MovieID"].astype(int)
        known_mask = movie_id_series.isin(self.movie_id_to_index)
        filtered = ratings[known_mask].copy()

        if filtered.empty:
            return {}

        # Vectorised: map MovieID → embedding index, Rating → rating index (0–4)
        filtered["_item_idx"] = filtered["MovieID"].map(self.movie_id_to_index).astype(int)
        filtered["_rating_idx"] = filtered["Rating"].clip(1, 5).astype(int) - 1

        # Sort globally by Timestamp descending — within each group the order is preserved.
        filtered = filtered.sort_values("Timestamp", ascending=False)

        histories: Dict[int, Dict[str, np.ndarray]] = {}

        for user_id, group in filtered.groupby("UserID"):
            user_id = int(user_id)
            # group is already sorted most-recent-first from the global sort.
            # Truncate to maximum_history_length.
            truncated = group.head(self.maximum_history_length)

            item_indices = truncated["_item_idx"].values
            rating_idx = truncated["_rating_idx"].values.astype(np.int64)

            # Gather embeddings using advanced indexing (vectorised).
            histories[user_id] = {
                "embeddings": self.raw_embeddings[item_indices],  # (L, 1536)
                "rating_indices": rating_idx,                     # (L,)
            }

        return histories

    @torch.no_grad()
    def _predict_batch(
        self,
        batch_user_ids: list[int],
        user_histories: Dict[int, Dict[str, np.ndarray]],
        users_dataframe: pd.DataFrame,
        seen_by_user: Dict[int, set[int]],
        k: int,
    ) -> Dict[int, List[Rating]]:
        """Encode a batch of users and retrieve top-K items from FAISS."""
        self.user_tower.eval()

        # Separate users with history from cold-start users.
        users_with_history: list[int] = []
        cold_users: list[int] = []
        for user_id in batch_user_ids:
            if user_id in user_histories:
                users_with_history.append(user_id)
            else:
                cold_users.append(user_id)

        predictions: Dict[int, List[Rating]] = {}

        # ── Cold-start users: popularity fallback ────────────────────
        for user_id in cold_users:
            predictions[user_id] = self._popularity_fallback(
                seen=seen_by_user.get(user_id, set()),
                k=k,
            )

        if not users_with_history:
            return predictions

        # ── Build batch tensors for users with history ───────────────
        # Use the pre-indexed lookup if available, otherwise build it once.
        if users_dataframe.index.name == "UserID":
            users_lookup = users_dataframe
        else:
            users_lookup = users_dataframe.set_index("UserID")

        max_history_length = max(
            len(user_histories[uid]["rating_indices"])
            for uid in users_with_history
        )
        actual_batch_size = len(users_with_history)

        history_embeddings = np.zeros(
            (actual_batch_size, max_history_length, self.item_embedding_dimension),
            dtype=np.float32,
        )
        rating_indices = np.zeros(
            (actual_batch_size, max_history_length), dtype=np.int64,
        )
        padding_mask = np.ones(
            (actual_batch_size, max_history_length), dtype=bool,
        )  # True = padded (invalid)

        gender_indices = np.zeros(actual_batch_size, dtype=np.int64)
        age_indices = np.zeros(actual_batch_size, dtype=np.int64)
        occupation_indices = np.zeros(actual_batch_size, dtype=np.int64)

        for batch_position, user_id in enumerate(users_with_history):
            history = user_histories[user_id]
            history_length = len(history["rating_indices"])

            history_embeddings[batch_position, :history_length] = (
                history["embeddings"]
            )
            rating_indices[batch_position, :history_length] = (
                history["rating_indices"]
            )
            padding_mask[batch_position, :history_length] = False

            if user_id in users_lookup.index:
                user_row = users_lookup.loc[user_id]
                gender_indices[batch_position] = GENDER_TO_INDEX.get(
                    user_row["Gender"], 0,
                )
                age_indices[batch_position] = AGE_BUCKET_TO_INDEX.get(
                    int(user_row["Age"]), 0,
                )
                occupation_indices[batch_position] = int(
                    user_row["Occupation"],
                )

        # ── Forward pass through user tower ──────────────────────────
        user_embeddings = self.user_tower(
            item_history_embeddings=torch.from_numpy(history_embeddings).to(
                self.device,
            ),
            history_rating_indices=torch.from_numpy(rating_indices).to(
                self.device,
            ),
            history_padding_mask=torch.from_numpy(padding_mask).to(
                self.device,
            ),
            gender_indices=torch.from_numpy(gender_indices).to(self.device),
            age_indices=torch.from_numpy(age_indices).to(self.device),
            occupation_indices=torch.from_numpy(occupation_indices).to(
                self.device,
            ),
        )
        user_embeddings_numpy = np.ascontiguousarray(
            user_embeddings.cpu().numpy().astype(np.float32),
        )

        # ── FAISS retrieval ──────────────────────────────────────────
        similarities, faiss_indices = self.faiss_index.search(
            user_embeddings_numpy,
            self.faiss_search_candidates,
        )

        for batch_position, user_id in enumerate(users_with_history):
            seen = seen_by_user.get(user_id, set())
            user_ratings: List[Rating] = []

            for rank in range(self.faiss_search_candidates):
                item_faiss_index = int(faiss_indices[batch_position, rank])
                if item_faiss_index < 0:
                    continue
                movie_id = self.index_to_movie_id.get(item_faiss_index)
                if movie_id is None or movie_id in seen:
                    continue
                user_ratings.append(
                    Rating(
                        movie_id=movie_id,
                        score=float(similarities[batch_position, rank]),
                    )
                )
                if len(user_ratings) >= k:
                    break

            predictions[user_id] = user_ratings

        return predictions

    def _popularity_fallback(
        self,
        seen: set[int],
        k: int,
    ) -> List[Rating]:
        """Return top-k globally popular movies the user hasn't seen."""
        if self._popularity_movie_ids is None or len(self._popularity_movie_ids) == 0:
            return []

        result: List[Rating] = []
        for movie_id, score in zip(
            self._popularity_movie_ids, self._popularity_scores,
        ):
            movie_id = int(movie_id)
            if movie_id not in seen:
                result.append(Rating(movie_id=movie_id, score=float(score)))
                if len(result) >= k:
                    break
        return result

    # ─────────────────────────────────────────────────────────────────────
    # Checkpoint management
    # ─────────────────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str) -> None:
        """Save model weights and configuration to a file.

        Parameters
        ----------
        path : str
            File path for the checkpoint (``.pt``).
        """
        torch.save(
            {
                "item_tower_state_dict": self.item_tower.state_dict(),
                "user_tower_state_dict": self.user_tower.state_dict(),
                "movie_id_to_index": self.movie_id_to_index,
                "index_to_movie_id": self.index_to_movie_id,
                "config": {
                    "projection_dimension": self.projection_dimension,
                    "item_embedding_dimension": self.item_embedding_dimension,
                    "maximum_history_length": self.maximum_history_length,
                    "positive_threshold": self.positive_threshold,
                    "faiss_search_candidates": self.faiss_search_candidates,
                    "inference_batch_size": self.inference_batch_size,
                },
            },
            path,
        )
        logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: str) -> None:
        """Load model weights from a previously saved checkpoint.

        Parameters
        ----------
        path : str
            File path for the checkpoint (``.pt``).
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.item_tower.load_state_dict(checkpoint["item_tower_state_dict"])
        self.user_tower.load_state_dict(checkpoint["user_tower_state_dict"])
        self.movie_id_to_index = checkpoint["movie_id_to_index"]
        self.index_to_movie_id = checkpoint["index_to_movie_id"]

        # Invalidate any stale FAISS index.
        self.faiss_index = None

        self.item_tower.to(self.device)
        self.user_tower.to(self.device)
        logger.info("Checkpoint loaded from %s", path)

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return all trainable parameters from both towers."""
        return list(self.item_tower.parameters()) + list(
            self.user_tower.parameters()
        )

    def train_mode(self) -> None:
        """Set both towers to training mode."""
        self.item_tower.train()
        self.user_tower.train()

    def eval_mode(self) -> None:
        """Set both towers to evaluation mode."""
        self.item_tower.eval()
        self.user_tower.eval()
