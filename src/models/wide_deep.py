from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.base import Rating, RecommenderModel

logger = logging.getLogger(__name__)


def _detect_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


class _WideAndDeepNet(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_genders: int,
        n_ages: int,
        n_occupations: int,
        n_genres: int,
        embedding_dim: int,
        hidden_dims: tuple[int, int],
        dropout: float,
        genre_embedding_dim: int,
    ) -> None:
        super().__init__()

        # wide part (memorization via sparse linear terms)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.wide_user_bias = nn.Embedding(n_users, 1)
        self.wide_item_bias = nn.Embedding(n_items, 1)
        self.wide_gender_bias = nn.Embedding(n_genders, 1)
        self.wide_age_bias = nn.Embedding(n_ages, 1)
        self.wide_occupation_bias = nn.Embedding(n_occupations, 1)
        self.wide_genre_linear = nn.Linear(n_genres, 1, bias=False)

        # deep part (generalization through dense embeddings + MLP)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.gender_embedding = nn.Embedding(n_genders, max(4, embedding_dim // 4))
        self.age_embedding = nn.Embedding(n_ages, max(4, embedding_dim // 4))
        self.occupation_embedding = nn.Embedding(
            n_occupations,
            max(4, embedding_dim // 4),
        )
        self.genre_projection = nn.Linear(n_genres, genre_embedding_dim)

        deep_input_dim = (
            embedding_dim
            + embedding_dim
            + max(4, embedding_dim // 4)
            + max(4, embedding_dim // 4)
            + max(4, embedding_dim // 4)
            + genre_embedding_dim
        )
        self.deep_mlp = nn.Sequential(
            nn.Linear(deep_input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        gender_idx: torch.Tensor,
        age_idx: torch.Tensor,
        occupation_idx: torch.Tensor,
        item_genre_features: torch.Tensor,
    ) -> torch.Tensor:
        wide_logit = (
            self.global_bias
            + self.wide_user_bias(user_idx).squeeze(-1)
            + self.wide_item_bias(item_idx).squeeze(-1)
            + self.wide_gender_bias(gender_idx).squeeze(-1)
            + self.wide_age_bias(age_idx).squeeze(-1)
            + self.wide_occupation_bias(occupation_idx).squeeze(-1)
            + self.wide_genre_linear(item_genre_features).squeeze(-1)
        )

        deep_input = torch.cat(
            [
                self.user_embedding(user_idx),
                self.item_embedding(item_idx),
                self.gender_embedding(gender_idx),
                self.age_embedding(age_idx),
                self.occupation_embedding(occupation_idx),
                self.genre_projection(item_genre_features),
            ],
            dim=1,
        )
        deep_logit = self.deep_mlp(deep_input).squeeze(-1)
        return wide_logit + deep_logit


class WideAndDeepRecommender(RecommenderModel):
    """Classical Wide & Deep recommender for implicit ranking.

    Training objective:
    - Binary cross-entropy over positive (rating >= threshold) interactions
      and sampled negatives.
    """

    def __init__(
        self,
        n_epochs: int = 5,
        batch_size: int = 1024,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-6,
        threshold: float = 4.0,
        n_negatives: int = 2,
        embedding_dim: int = 64,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.2,
        genre_embedding_dim: int = 16,
        max_positive_samples_per_epoch: int = 0,
        gradient_clip_norm: float = 5.0,
        random_state: int = 42,
        device: str = "auto",
    ) -> None:
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.threshold = float(threshold)
        self.n_negatives = int(n_negatives)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dims = tuple(int(v) for v in hidden_dims)
        self.dropout = float(dropout)
        self.genre_embedding_dim = int(genre_embedding_dim)
        self.max_positive_samples_per_epoch = int(max_positive_samples_per_epoch)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.random_state = int(random_state)
        self.device = _detect_device(device)

        self.user_to_idx: Dict[int, int] = {}
        self.item_to_idx: Dict[int, int] = {}
        self.idx_to_item: Dict[int, int] = {}

        self.gender_to_idx: Dict[str, int] = {}
        self.age_to_idx: Dict[int, int] = {}
        self.occupation_to_idx: Dict[int, int] = {}

        self.user_gender_idx: np.ndarray | None = None
        self.user_age_idx: np.ndarray | None = None
        self.user_occupation_idx: np.ndarray | None = None
        self.item_genre_matrix: np.ndarray | None = None

        self._user_seen_idx_set: Dict[int, set[int]] = {}
        self._all_item_indices: np.ndarray = np.array([], dtype=np.int64)
        self._positive_pairs: np.ndarray = np.zeros((0, 2), dtype=np.int64)

        self._global_popularity_scores: np.ndarray = np.array([], dtype=np.float64)
        self.loss_history_: List[float] = []
        self.val_history_: List[dict] = []
        self.best_epoch_: int = -1
        self.best_val_ndcg_: float = float("-inf")
        self.total_training_time_seconds_: float = 0.0

        self.model: _WideAndDeepNet | None = None

    @staticmethod
    def _resolve_item_columns(movies: pd.DataFrame) -> tuple[str, str]:
        if "MovieID" in movies.columns:
            movie_col = "MovieID"
            genre_col = "Genres" if "Genres" in movies.columns else ""
            return movie_col, genre_col
        if "movie_id" in movies.columns:
            movie_col = "movie_id"
            genre_col = "genres" if "genres" in movies.columns else ""
            return movie_col, genre_col
        raise ValueError("movies dataframe must contain MovieID or movie_id")

    def _build_mappings(self, users: pd.DataFrame, movies: pd.DataFrame) -> None:
        user_ids = users["UserID"].drop_duplicates().astype(int).to_numpy()
        movie_col, _ = self._resolve_item_columns(movies)
        item_ids = movies[movie_col].drop_duplicates().astype(int).to_numpy()

        self.user_to_idx = {uid: idx for idx, uid in enumerate(user_ids.tolist())}
        self.item_to_idx = {mid: idx for idx, mid in enumerate(item_ids.tolist())}
        self.idx_to_item = {idx: mid for mid, idx in self.item_to_idx.items()}
        self._all_item_indices = np.arange(len(item_ids), dtype=np.int64)

    def _build_user_features(self, users: pd.DataFrame) -> None:
        genders = users["Gender"].fillna("UNK").astype(str).unique().tolist()
        ages = users["Age"].fillna(-1).astype(int).unique().tolist()
        occupations = users["Occupation"].fillna(-1).astype(int).unique().tolist()

        self.gender_to_idx = {value: idx for idx, value in enumerate(sorted(genders))}
        self.age_to_idx = {value: idx for idx, value in enumerate(sorted(ages))}
        self.occupation_to_idx = {
            value: idx for idx, value in enumerate(sorted(occupations))
        }

        n_users = len(self.user_to_idx)
        self.user_gender_idx = np.zeros(n_users, dtype=np.int64)
        self.user_age_idx = np.zeros(n_users, dtype=np.int64)
        self.user_occupation_idx = np.zeros(n_users, dtype=np.int64)

        for _, row in users.iterrows():
            uid = int(row["UserID"])
            u_idx = self.user_to_idx.get(uid)
            if u_idx is None:
                continue
            self.user_gender_idx[u_idx] = self.gender_to_idx[str(row["Gender"])]
            self.user_age_idx[u_idx] = self.age_to_idx[int(row["Age"])]
            self.user_occupation_idx[u_idx] = self.occupation_to_idx[
                int(row["Occupation"])
            ]

    def _build_item_features(self, movies: pd.DataFrame) -> None:
        movie_col, genre_col = self._resolve_item_columns(movies)

        n_items = len(self.item_to_idx)
        if not genre_col:
            self.item_genre_matrix = np.zeros((n_items, 1), dtype=np.float32)
            return

        genre_lists: Dict[int, List[str]] = {}
        vocab_set: set[str] = set()
        for _, row in movies[[movie_col, genre_col]].iterrows():
            mid = int(row[movie_col])
            if mid not in self.item_to_idx:
                continue
            genres_raw = str(row[genre_col]) if pd.notna(row[genre_col]) else ""
            genres = [g.strip() for g in genres_raw.split("|") if g.strip()]
            if not genres:
                genres = ["__UNKNOWN__"]
            genre_lists[mid] = genres
            vocab_set.update(genres)

        vocab = sorted(vocab_set) if vocab_set else ["__UNKNOWN__"]
        genre_to_idx = {g: i for i, g in enumerate(vocab)}
        matrix = np.zeros((n_items, len(vocab)), dtype=np.float32)
        for mid, genres in genre_lists.items():
            i_idx = self.item_to_idx[mid]
            for genre in genres:
                matrix[i_idx, genre_to_idx[genre]] = 1.0
        self.item_genre_matrix = matrix

    def _build_positive_pairs(self, ratings: pd.DataFrame) -> None:
        positives = ratings[ratings["Rating"] >= self.threshold].copy()
        if positives.empty:
            logger.warning(
                "No ratings >= %.2f found; falling back to all interactions as positives.",
                self.threshold,
            )
            positives = ratings.copy()

        self._user_seen_idx_set = {}
        for uid, group in ratings.groupby("UserID"):
            uid = int(uid)
            u_idx = self.user_to_idx.get(uid)
            if u_idx is None:
                continue
            seen_idx = {
                self.item_to_idx[int(mid)]
                for mid in group["MovieID"].astype(int).tolist()
                if int(mid) in self.item_to_idx
            }
            self._user_seen_idx_set[u_idx] = seen_idx

        pairs = []
        for uid, group in positives.groupby("UserID"):
            uid = int(uid)
            u_idx = self.user_to_idx.get(uid)
            if u_idx is None:
                continue
            for mid in group["MovieID"].astype(int).tolist():
                i_idx = self.item_to_idx.get(int(mid))
                if i_idx is not None:
                    pairs.append((u_idx, i_idx))

        if not pairs:
            raise ValueError("No positive training pairs available for WideAndDeep.")
        self._positive_pairs = np.array(pairs, dtype=np.int64)

    def _build_popularity_scores(self, ratings: pd.DataFrame) -> None:
        scores = np.zeros(len(self.item_to_idx), dtype=np.float64)
        for mid, count in ratings["MovieID"].value_counts().items():
            i_idx = self.item_to_idx.get(int(mid))
            if i_idx is not None:
                scores[i_idx] = float(count)
        self._global_popularity_scores = scores

    def _sample_negative(self, rng: np.random.Generator, seen_set: set[int]) -> int:
        while True:
            j = int(rng.integers(0, len(self._all_item_indices)))
            if j not in seen_set:
                return j

    def _build_epoch_samples(
        self,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos_pairs = self._positive_pairs
        if (
            self.max_positive_samples_per_epoch > 0
            and len(pos_pairs) > self.max_positive_samples_per_epoch
        ):
            select = rng.choice(
                len(pos_pairs),
                size=self.max_positive_samples_per_epoch,
                replace=False,
            )
            pos_pairs = pos_pairs[select]

        n_pos = len(pos_pairs)
        total = n_pos * (1 + self.n_negatives)
        users = np.zeros(total, dtype=np.int64)
        items = np.zeros(total, dtype=np.int64)
        labels = np.zeros(total, dtype=np.float32)

        ptr = 0
        for u_idx, i_idx in pos_pairs:
            users[ptr] = int(u_idx)
            items[ptr] = int(i_idx)
            labels[ptr] = 1.0
            ptr += 1

            seen = self._user_seen_idx_set.get(int(u_idx), set())
            if len(seen) >= len(self._all_item_indices):
                continue

            for _ in range(self.n_negatives):
                j_idx = self._sample_negative(rng, seen)
                users[ptr] = int(u_idx)
                items[ptr] = int(j_idx)
                labels[ptr] = 0.0
                ptr += 1

        return users[:ptr], items[:ptr], labels[:ptr]

    def fit(
        self,
        ratings: pd.DataFrame,
        users: pd.DataFrame | None = None,
        movies: pd.DataFrame | None = None,
        val_ratings: pd.DataFrame | None = None,
        eval_ks: tuple[int, ...] = (10, 20),
        monitor_k: int = 10,
        eval_mode: str = "all",
        save_best_model: bool = False,
        best_model_path: str | None = None,
        restore_best_weights: bool = True,
        early_stopping_patience: int = 0,
    ) -> "WideAndDeepRecommender":
        if users is None or users.empty:
            raise ValueError("users dataframe is required for WideAndDeepRecommender")
        if movies is None or movies.empty:
            raise ValueError("movies dataframe is required for WideAndDeepRecommender")
        if ratings.empty:
            raise ValueError("ratings dataframe is empty")

        self._build_mappings(users=users, movies=movies)
        self._build_user_features(users=users)
        self._build_item_features(movies=movies)
        self._build_positive_pairs(ratings=ratings)
        self._build_popularity_scores(ratings=ratings)

        assert self.user_gender_idx is not None
        assert self.user_age_idx is not None
        assert self.user_occupation_idx is not None
        assert self.item_genre_matrix is not None

        self.model = _WideAndDeepNet(
            n_users=len(self.user_to_idx),
            n_items=len(self.item_to_idx),
            n_genders=max(1, len(self.gender_to_idx)),
            n_ages=max(1, len(self.age_to_idx)),
            n_occupations=max(1, len(self.occupation_to_idx)),
            n_genres=self.item_genre_matrix.shape[1],
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            genre_embedding_dim=self.genre_embedding_dim,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.BCEWithLogitsLoss()

        rng = np.random.default_rng(self.random_state)
        self.loss_history_.clear()
        self.val_history_.clear()
        self.best_epoch_ = -1
        self.best_val_ndcg_ = float("-inf")
        best_state_dict = None
        epochs_without_improvement = 0
        training_start = time.perf_counter()

        logger.info(
            "WideAndDeep training started: epochs=%d, positives=%d, device=%s",
            self.n_epochs,
            len(self._positive_pairs),
            self.device,
        )
        for epoch in range(self.n_epochs):
            users_idx, items_idx, labels = self._build_epoch_samples(rng)
            order = np.arange(len(labels), dtype=np.int64)
            rng.shuffle(order)

            epoch_loss = 0.0
            n_batches = 0
            self.model.train()

            for start in range(0, len(order), self.batch_size):
                batch_idx = order[start : start + self.batch_size]
                batch_users = users_idx[batch_idx]
                batch_items = items_idx[batch_idx]

                batch_gender = self.user_gender_idx[batch_users]
                batch_age = self.user_age_idx[batch_users]
                batch_occ = self.user_occupation_idx[batch_users]
                batch_genres = self.item_genre_matrix[batch_items]

                user_t = torch.from_numpy(batch_users).to(
                    self.device, non_blocking=True
                )
                item_t = torch.from_numpy(batch_items).to(
                    self.device, non_blocking=True
                )
                gender_t = torch.from_numpy(batch_gender).to(
                    self.device, non_blocking=True
                )
                age_t = torch.from_numpy(batch_age).to(self.device, non_blocking=True)
                occ_t = torch.from_numpy(batch_occ).to(self.device, non_blocking=True)
                genres_t = torch.from_numpy(batch_genres).to(
                    self.device, non_blocking=True
                )
                labels_t = torch.from_numpy(labels[batch_idx]).to(
                    self.device,
                    non_blocking=True,
                )

                logits = self.model(
                    user_idx=user_t,
                    item_idx=item_t,
                    gender_idx=gender_t,
                    age_idx=age_t,
                    occupation_idx=occ_t,
                    item_genre_features=genres_t,
                )
                if not torch.isfinite(logits).all():
                    raise RuntimeError(
                        f"Non-finite logits detected at epoch={epoch + 1}, batch_start={start}"
                    )
                loss = loss_fn(logits, labels_t)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss detected at epoch={epoch + 1}, batch_start={start}"
                    )
                if float(loss.item()) < 0.0:
                    raise RuntimeError(
                        f"Negative BCE loss detected at epoch={epoch + 1}, batch_start={start}: {float(loss.item())}"
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.gradient_clip_norm,
                    )
                optimizer.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            avg_loss = epoch_loss / max(1, n_batches)
            self.loss_history_.append(avg_loss)
            logger.info(
                "WideAndDeep epoch %d/%d - avg BCE loss: %.6f",
                epoch + 1,
                self.n_epochs,
                avg_loss,
            )

            if val_ratings is not None:
                from src.eval.offline_ranking import evaluate as evaluate_offline

                report = evaluate_offline(
                    model=self,
                    train_ratings=ratings,
                    test_ratings=val_ratings,
                    users=users,
                    movies=movies,
                    ks=eval_ks,
                    threshold=self.threshold,
                    mode=eval_mode,
                )
                if monitor_k not in report.by_k:
                    raise ValueError(f"monitor_k={monitor_k} missing in eval_ks={eval_ks}")

                monitored_ndcg = float(report.by_k[monitor_k].ndcg)
                self.val_history_.append(
                    {"epoch": epoch + 1, "monitor_k": int(monitor_k), "ndcg": monitored_ndcg}
                )
                logger.info("  Val epoch %d - NDCG@%d: %.6f", epoch + 1, monitor_k, monitored_ndcg)

                improved = monitored_ndcg > self.best_val_ndcg_
                if improved:
                    self.best_val_ndcg_ = monitored_ndcg
                    self.best_epoch_ = epoch + 1
                    best_state_dict = copy.deepcopy(self.model.state_dict())
                    epochs_without_improvement = 0
                    if save_best_model and best_model_path:
                        self.save_checkpoint(best_model_path)
                else:
                    epochs_without_improvement += 1

                if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d). Best epoch=%d, best NDCG@%d=%.6f",
                        epoch + 1,
                        early_stopping_patience,
                        self.best_epoch_,
                        monitor_k,
                        self.best_val_ndcg_,
                    )
                    break

        self.total_training_time_seconds_ = time.perf_counter() - training_start
        logger.info("WideAndDeep training complete in %.2f seconds", self.total_training_time_seconds_)

        if restore_best_weights and best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            logger.info(
                "Restored best weights from epoch %d (NDCG@%d=%.6f)",
                self.best_epoch_,
                monitor_k,
                self.best_val_ndcg_,
            )

        return self

    def save_checkpoint(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Model is not initialised; call fit() first.")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_config": {
                    "n_users": len(self.user_to_idx),
                    "n_items": len(self.item_to_idx),
                    "n_genders": max(1, len(self.gender_to_idx)),
                    "n_ages": max(1, len(self.age_to_idx)),
                    "n_occupations": max(1, len(self.occupation_to_idx)),
                    "n_genres": int(self.item_genre_matrix.shape[1]) if self.item_genre_matrix is not None else 1,
                    "embedding_dim": self.embedding_dim,
                    "hidden_dims": self.hidden_dims,
                    "dropout": self.dropout,
                    "genre_embedding_dim": self.genre_embedding_dim,
                },
                "mappings": {
                    "user_to_idx": self.user_to_idx,
                    "item_to_idx": self.item_to_idx,
                    "idx_to_item": self.idx_to_item,
                    "gender_to_idx": self.gender_to_idx,
                    "age_to_idx": self.age_to_idx,
                    "occupation_to_idx": self.occupation_to_idx,
                },
                "arrays": {
                    "user_gender_idx": self.user_gender_idx,
                    "user_age_idx": self.user_age_idx,
                    "user_occupation_idx": self.user_occupation_idx,
                    "item_genre_matrix": self.item_genre_matrix,
                    "all_item_indices": self._all_item_indices,
                    "global_popularity_scores": self._global_popularity_scores,
                },
                "stats": {
                    "loss_history": self.loss_history_,
                    "val_history": self.val_history_,
                    "best_epoch": self.best_epoch_,
                    "best_val_ndcg": self.best_val_ndcg_,
                    "total_training_time_seconds": self.total_training_time_seconds_,
                },
            },
            str(p),
        )
        logger.info("Saved WideAndDeep checkpoint to %s", p)

    def load_checkpoint(self, path: str, map_location: str | torch.device | None = None) -> None:
        ckpt = torch.load(path, map_location=map_location or self.device, weights_only=False)
        cfg = ckpt["model_config"]
        self.model = _WideAndDeepNet(**cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        maps = ckpt["mappings"]
        self.user_to_idx = maps["user_to_idx"]
        self.item_to_idx = maps["item_to_idx"]
        self.idx_to_item = maps["idx_to_item"]
        self.gender_to_idx = maps["gender_to_idx"]
        self.age_to_idx = maps["age_to_idx"]
        self.occupation_to_idx = maps["occupation_to_idx"]

        arr = ckpt["arrays"]
        self.user_gender_idx = arr["user_gender_idx"]
        self.user_age_idx = arr["user_age_idx"]
        self.user_occupation_idx = arr["user_occupation_idx"]
        self.item_genre_matrix = arr["item_genre_matrix"]
        self._all_item_indices = arr["all_item_indices"]
        self._global_popularity_scores = arr["global_popularity_scores"]

        stats = ckpt.get("stats", {})
        self.loss_history_ = list(stats.get("loss_history", []))
        self.val_history_ = list(stats.get("val_history", []))
        self.best_epoch_ = int(stats.get("best_epoch", -1))
        self.best_val_ndcg_ = float(stats.get("best_val_ndcg", float("-inf")))
        self.total_training_time_seconds_ = float(stats.get("total_training_time_seconds", 0.0))
        logger.info("Loaded WideAndDeep checkpoint from %s", path)

    def _score_items_for_user(
        self,
        u_idx: int,
        candidate_indices: np.ndarray,
    ) -> np.ndarray:
        assert self.model is not None
        assert self.user_gender_idx is not None
        assert self.user_age_idx is not None
        assert self.user_occupation_idx is not None
        assert self.item_genre_matrix is not None

        self.model.eval()
        out_scores = np.zeros(len(candidate_indices), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(candidate_indices), self.batch_size):
                batch_items = candidate_indices[start : start + self.batch_size]
                batch_size = len(batch_items)
                batch_users = np.full(batch_size, u_idx, dtype=np.int64)
                batch_gender = np.full(
                    batch_size,
                    self.user_gender_idx[u_idx],
                    dtype=np.int64,
                )
                batch_age = np.full(
                    batch_size, self.user_age_idx[u_idx], dtype=np.int64
                )
                batch_occ = np.full(
                    batch_size,
                    self.user_occupation_idx[u_idx],
                    dtype=np.int64,
                )
                batch_genres = self.item_genre_matrix[batch_items]

                logits = self.model(
                    user_idx=torch.from_numpy(batch_users).to(
                        self.device, non_blocking=True
                    ),
                    item_idx=torch.from_numpy(batch_items).to(
                        self.device, non_blocking=True
                    ),
                    gender_idx=torch.from_numpy(batch_gender).to(
                        self.device, non_blocking=True
                    ),
                    age_idx=torch.from_numpy(batch_age).to(
                        self.device, non_blocking=True
                    ),
                    occupation_idx=torch.from_numpy(batch_occ).to(
                        self.device,
                        non_blocking=True,
                    ),
                    item_genre_features=torch.from_numpy(batch_genres).to(
                        self.device,
                        non_blocking=True,
                    ),
                )
                out_scores[start : start + batch_size] = (
                    logits.detach().cpu().numpy().astype(np.float32)
                )
        return out_scores

    def predict(
        self,
        users: pd.DataFrame,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        k: int = 10,
    ) -> Dict[int, List[Rating]]:
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit(...) first.")

        if "MovieID" in movies.columns:
            catalog_ids = set(movies["MovieID"].astype(int).tolist())
        elif "movie_id" in movies.columns:
            catalog_ids = set(movies["movie_id"].astype(int).tolist())
        else:
            catalog_ids = set(self.item_to_idx.keys())

        candidate_indices = np.array(
            [idx for mid, idx in self.item_to_idx.items() if mid in catalog_ids],
            dtype=np.int64,
        )
        if candidate_indices.size == 0:
            return {int(uid): [] for uid in users["UserID"].astype(int).values}

        seen_by_user = {
            int(uid): set(group["MovieID"].astype(int).tolist())
            for uid, group in ratings.groupby("UserID")
        }

        item_ids = np.array(
            [int(self.idx_to_item[i]) for i in candidate_indices], dtype=np.int64
        )
        preds: Dict[int, List[Rating]] = {}

        for uid in users["UserID"].astype(int).values:
            uid = int(uid)
            seen = seen_by_user.get(uid, set())

            if uid in self.user_to_idx:
                u_idx = self.user_to_idx[uid]
                scores = self._score_items_for_user(
                    u_idx=u_idx, candidate_indices=candidate_indices
                )
            else:
                scores = self._global_popularity_scores[candidate_indices].astype(
                    np.float32
                )

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
