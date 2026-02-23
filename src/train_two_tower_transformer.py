"""Train and evaluate the Two-Tower Transformer recommender.

Creates a run directory under ``./runs/`` with checkpoints, a training log,
and a post-training Markdown report.

Usage examples
--------------
Default training (30 epochs):
    python -m src.train_two_tower_transformer

Quick smoke test:
    python -m src.train_two_tower_transformer --number-of-epochs 2 \\
        --evaluation-interval 1 --batch-size 64

Custom hyperparameters:
    python -m src.train_two_tower_transformer --number-of-epochs 50 \\
        --learning-rate 5e-4 --temperature 0.07

Resume from a checkpoint:
    python -m src.train_two_tower_transformer \\
        --resume-from runs/two_tower_20250101_120000/checkpoints/best.pt

See all options:
    python -m src.train_two_tower_transformer --help
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Prevent OpenMP conflict between torch and faiss-cpu on macOS.
# Both ship their own libomp.dylib; this allows coexistence.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from data.dataframes import (
    movies_enriched,
    user_based_temporal_train,
    user_based_temporal_val,
    users,
)
from src.eval.eval import evaluate as evaluate_basic
from src.eval.offline_ranking import evaluate as evaluate_offline
from src.models.ANN.two_towers.two_tower_transformer import (
    TwoTowerTransformerRecommender,
)
from src.models.ANN.two_towers.user_tower import (
    AGE_BUCKET_TO_INDEX,
    GENDER_TO_INDEX,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────


class InteractionDataset(Dataset):
    """PyTorch dataset of (user, positive_item) training pairs.

    For each positive interaction (rating >= threshold) at time *t*, the
    history is all preceding interactions (any rating), sorted most-recent
    first, truncated to ``maximum_history_length``.  The target item is
    excluded from the history to prevent trivial shortcutting.

    Interactions with empty preceding history (the very first interaction per
    user) are skipped.
    """

    def __init__(
        self,
        ratings: pd.DataFrame,
        users_dataframe: pd.DataFrame,
        raw_embeddings: np.ndarray,
        movie_id_to_index: Dict[int, int],
        positive_threshold: float = 4.0,
        maximum_history_length: int = 64,
    ) -> None:
        super().__init__()

        self.raw_embeddings = raw_embeddings.astype(np.float32)
        self.movie_id_to_index = movie_id_to_index
        self.maximum_history_length = maximum_history_length

        # Build user demographics lookup.
        self.user_demographics: Dict[int, Dict[str, int]] = {}
        for _, row in users_dataframe.iterrows():
            user_id = int(row["UserID"])
            self.user_demographics[user_id] = {
                "gender_index": GENDER_TO_INDEX.get(row["Gender"], 0),
                "age_index": AGE_BUCKET_TO_INDEX.get(int(row["Age"]), 0),
                "occupation_index": int(row["Occupation"]),
            }

        # Pre-compute all training samples.
        self.samples: List[Dict] = []
        self._build_samples(ratings, positive_threshold)
        logger.info(
            "InteractionDataset: %d samples from %d positive interactions",
            len(self.samples),
            len(ratings[ratings["Rating"] >= positive_threshold]),
        )

    def _build_samples(
        self,
        ratings: pd.DataFrame,
        positive_threshold: float,
    ) -> None:
        """Pre-compute training samples from the ratings dataframe."""
        for user_id, group in ratings.groupby("UserID"):
            user_id = int(user_id)
            sorted_group = group.sort_values("Timestamp").reset_index(drop=True)

            # Track the running history as we walk forward in time.
            history_movie_indices: List[int] = []
            history_rating_indices: List[int] = []

            for _, row in sorted_group.iterrows():
                movie_id = int(row["MovieID"])
                if movie_id not in self.movie_id_to_index:
                    continue

                item_index = self.movie_id_to_index[movie_id]
                rating = int(row["Rating"])
                rating_index = max(0, min(4, rating - 1))  # 1–5 → 0–4

                # If this is a positive interaction and there is preceding
                # history, create a training sample.
                if rating >= positive_threshold and len(history_movie_indices) > 0:
                    # Take the most recent items (end of the lists).
                    history_length = min(
                        len(history_movie_indices),
                        self.maximum_history_length,
                    )
                    # Reverse so that index 0 = most recent item.
                    recent_movie_indices = list(
                        reversed(history_movie_indices[-history_length:])
                    )
                    recent_rating_indices = list(
                        reversed(history_rating_indices[-history_length:])
                    )

                    demographics = self.user_demographics.get(
                        user_id,
                        {"gender_index": 0, "age_index": 0, "occupation_index": 0},
                    )

                    self.samples.append(
                        {
                            "positive_item_index": item_index,
                            "history_movie_indices": np.array(
                                recent_movie_indices, dtype=np.int64,
                            ),
                            "history_rating_indices": np.array(
                                recent_rating_indices, dtype=np.int64,
                            ),
                            "history_length": history_length,
                            "gender_index": demographics["gender_index"],
                            "age_index": demographics["age_index"],
                            "occupation_index": demographics["occupation_index"],
                        }
                    )

                # Always append to history (regardless of rating).
                history_movie_indices.append(item_index)
                history_rating_indices.append(rating_index)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]

        # Look up the actual embeddings from pre-stored indices.
        positive_item_embedding = self.raw_embeddings[
            sample["positive_item_index"]
        ]
        history_item_embeddings = self.raw_embeddings[
            sample["history_movie_indices"]
        ]

        return {
            "positive_item_embedding": positive_item_embedding,
            "history_item_embeddings": history_item_embeddings,
            "history_rating_indices": sample["history_rating_indices"],
            "history_length": sample["history_length"],
            "gender_index": sample["gender_index"],
            "age_index": sample["age_index"],
            "occupation_index": sample["occupation_index"],
        }


def collate_interaction_batch(
    batch: List[Dict],
) -> Dict[str, torch.Tensor]:
    """Collate variable-length interaction samples into padded batch tensors.

    Histories are right-padded to the maximum length in the batch.  The
    padding mask has ``True`` at padded positions (PyTorch convention for
    ``src_key_padding_mask``).
    """
    batch_size = len(batch)
    max_history_length = max(sample["history_length"] for sample in batch)
    embedding_dimension = batch[0]["positive_item_embedding"].shape[0]

    positive_item_embeddings = np.stack(
        [sample["positive_item_embedding"] for sample in batch],
    )

    history_item_embeddings = np.zeros(
        (batch_size, max_history_length, embedding_dimension),
        dtype=np.float32,
    )
    history_rating_indices = np.zeros(
        (batch_size, max_history_length),
        dtype=np.int64,
    )
    padding_mask = np.ones(
        (batch_size, max_history_length),
        dtype=bool,
    )

    gender_indices = np.zeros(batch_size, dtype=np.int64)
    age_indices = np.zeros(batch_size, dtype=np.int64)
    occupation_indices = np.zeros(batch_size, dtype=np.int64)

    for position, sample in enumerate(batch):
        length = sample["history_length"]
        history_item_embeddings[position, :length] = sample[
            "history_item_embeddings"
        ]
        history_rating_indices[position, :length] = sample[
            "history_rating_indices"
        ]
        padding_mask[position, :length] = False

        gender_indices[position] = sample["gender_index"]
        age_indices[position] = sample["age_index"]
        occupation_indices[position] = sample["occupation_index"]

    return {
        "positive_item_embeddings": torch.from_numpy(positive_item_embeddings),
        "history_item_embeddings": torch.from_numpy(history_item_embeddings),
        "history_rating_indices": torch.from_numpy(history_rating_indices),
        "history_padding_mask": torch.from_numpy(padding_mask),
        "gender_indices": torch.from_numpy(gender_indices),
        "age_indices": torch.from_numpy(age_indices),
        "occupation_indices": torch.from_numpy(occupation_indices),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────


def _save_training_state(
    path: Path,
    model: TwoTowerTransformerRecommender,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_ndcg: float,
    best_epoch: int,
    epoch_records: List[Dict],
) -> None:
    """Persist the full training state so that training can be resumed exactly.

    Saves model weights, optimizer momentum buffers, LR scheduler counters,
    and bookkeeping (current epoch, best metric, history).
    """
    model.save_checkpoint(str(path))

    # Append training-loop state into the same file.
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    checkpoint["epoch"] = epoch
    checkpoint["best_ndcg"] = best_ndcg
    checkpoint["best_epoch"] = best_epoch
    checkpoint["epoch_records"] = epoch_records
    torch.save(checkpoint, str(path))


def _load_training_state(
    path: Path,
    model: TwoTowerTransformerRecommender,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> Dict:
    """Restore the full training state from a checkpoint.

    Returns a dict with ``epoch``, ``best_ndcg``, ``best_epoch``, and
    ``epoch_records`` so the caller can continue the loop from the right
    place.
    """
    model.load_checkpoint(str(path))

    checkpoint = torch.load(str(path), map_location=model.device, weights_only=False)
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "best_ndcg": checkpoint.get("best_ndcg", -1.0),
        "best_epoch": checkpoint.get("best_epoch", -1),
        "epoch_records": checkpoint.get("epoch_records", []),
    }


# Thermal throttling detection.
# On Apple-silicon Macs the MPS GPU throttles aggressively once it overheats.
# An epoch that takes more than ``_THROTTLE_FACTOR`` times the rolling median
# duration is considered throttled.  When detected the loop pauses for
# ``_COOLDOWN_SECONDS`` to let the hardware cool down before continuing.
_THROTTLE_FACTOR = 2.0
_COOLDOWN_SECONDS = 120  # 2 minutes


def train(
    model: TwoTowerTransformerRecommender,
    train_dataloader: DataLoader,
    validation_ratings: pd.DataFrame,
    users_dataframe: pd.DataFrame,
    movies_dataframe: pd.DataFrame,
    number_of_epochs: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    evaluation_interval: int,
    run_directory: Path,
    evaluation_ks: list[int],
    train_ratings: pd.DataFrame,
    resume_from: str | None = None,
) -> Dict:
    """Run the full training loop with periodic evaluation and checkpointing.

    Parameters
    ----------
    model : TwoTowerTransformerRecommender
        Prepared model (``prepare()`` already called).
    train_dataloader : DataLoader
        DataLoader yielding batches from ``InteractionDataset``.
    validation_ratings : pd.DataFrame
        Held-out ratings for periodic evaluation.
    users_dataframe : pd.DataFrame
        User side-information.
    movies_dataframe : pd.DataFrame
        Movie side-information (must contain ``movie_id`` column).
    number_of_epochs : int
        Total training epochs.
    learning_rate : float
        Peak learning rate for AdamW.
    weight_decay : float
        L2 regularisation strength.
    temperature : float
        Temperature parameter τ for InfoNCE loss.
    evaluation_interval : int
        Evaluate every N epochs.
    run_directory : Path
        Run directory for checkpoints and logs.
    evaluation_ks : list[int]
        K values for the offline evaluator in the final report.
    train_ratings : pd.DataFrame
        Training ratings (passed to ``model.predict()`` at eval time).
    resume_from : str | None
        Path to a full-state checkpoint to resume training from.

    Returns
    -------
    dict
        Training report data (losses, metrics, timings).
    """
    device = model.device
    all_parameters = model.trainable_parameters()

    optimizer = torch.optim.AdamW(
        all_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Linear warmup (10% of steps) + cosine decay to a minimum LR floor.
    # The floor prevents the learning rate from dropping to zero, which would
    # stall learning in the final epochs.
    total_steps = number_of_epochs * len(train_dataloader)
    warmup_steps = max(1, int(0.1 * total_steps))
    minimum_lr_fraction = 0.01  # LR floor = 1% of peak LR

    def learning_rate_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup from 0 to 1.
            return float(current_step) / float(warmup_steps)
        # Cosine decay from 1 → minimum_lr_fraction (not all the way to 0).
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return minimum_lr_fraction + (1.0 - minimum_lr_fraction) * cosine_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=learning_rate_lambda,
    )

    # ── Resume from checkpoint if requested ──────────────────────────
    best_ndcg = -1.0
    best_epoch = -1
    start_epoch = 0
    epoch_records: List[Dict] = []

    if resume_from and Path(resume_from).exists():
        logger.info("Resuming training state from %s", resume_from)
        state = _load_training_state(
            Path(resume_from), model, optimizer, scheduler,
        )
        start_epoch = state["epoch"] + 1  # Continue from the next epoch.
        best_ndcg = state["best_ndcg"]
        best_epoch = state["best_epoch"]
        epoch_records = state["epoch_records"]
        logger.info(
            "Resumed at epoch %d (best NDCG@10: %.5f at epoch %d)",
            start_epoch + 1,
            best_ndcg,
            best_epoch,
        )

    training_start_time = time.time()

    # Rolling window of recent epoch durations for throttle detection.
    epoch_durations: List[float] = []

    for epoch in range(start_epoch, number_of_epochs):
        epoch_start_time = time.time()
        model.train_mode()
        epoch_loss = 0.0
        number_of_batches = 0

        for batch in train_dataloader:
            # Move tensors to device.
            positive_item_embeddings = batch["positive_item_embeddings"].to(device)
            history_item_embeddings = batch["history_item_embeddings"].to(device)
            history_rating_indices = batch["history_rating_indices"].to(device)
            history_padding_mask = batch["history_padding_mask"].to(device)
            gender_indices = batch["gender_indices"].to(device)
            age_indices = batch["age_indices"].to(device)
            occupation_indices = batch["occupation_indices"].to(device)

            # Forward pass: Item tower
            # z_items: (B, projection_dimension), L2-normalised
            z_items = model.item_tower(positive_item_embeddings)

            # Forward pass: User tower
            # z_users: (B, projection_dimension), L2-normalised
            z_users = model.user_tower(
                item_history_embeddings=history_item_embeddings,
                history_rating_indices=history_rating_indices,
                history_padding_mask=history_padding_mask,
                gender_indices=gender_indices,
                age_indices=age_indices,
                occupation_indices=occupation_indices,
            )

            # InfoNCE loss with in-batch negatives.
            #
            # similarity_matrix[i, j] = z_user_i · z_item_j / τ
            # The diagonal (i == j) is the true positive pair.
            # Off-diagonal entries are in-batch negatives.
            #
            # L = -(1/B) Σ_i log(
            #     exp(z_user_i · z_item_i / τ)
            #     / Σ_j exp(z_user_i · z_item_j / τ)
            # )
            #
            # This is equivalent to cross-entropy with labels = [0, 1, ..., B-1].
            current_batch_size = z_users.shape[0]
            similarity_matrix = torch.matmul(z_users, z_items.T) / temperature
            labels = torch.arange(current_batch_size, device=device)
            loss = functional.cross_entropy(similarity_matrix, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            number_of_batches += 1

        average_epoch_loss = epoch_loss / max(number_of_batches, 1)
        current_learning_rate = scheduler.get_last_lr()[0]

        epoch_record: Dict = {
            "epoch": epoch + 1,
            "loss": average_epoch_loss,
            "learning_rate": current_learning_rate,
            "metrics": None,
        }

        logger.info(
            "Epoch %d/%d — loss: %.6f, lr: %.2e",
            epoch + 1,
            number_of_epochs,
            average_epoch_loss,
            current_learning_rate,
        )

        # Periodic evaluation.
        if (epoch + 1) % evaluation_interval == 0 or (epoch + 1) == number_of_epochs:
            model.eval_mode()
            model.faiss_index = None  # Force rebuild with updated weights.
            model.build_faiss_index()

            metrics = evaluate_basic(
                model=model,
                train_ratings=train_ratings,
                test_ratings=validation_ratings,
                users=users_dataframe,
                movies=movies_dataframe,
                k=10,
                threshold=model.positive_threshold,
            )

            epoch_record["metrics"] = {
                "ndcg_at_10": metrics.ndcg,
                "precision_at_10": metrics.precision,
                "recall_at_10": metrics.recall,
            }

            logger.info(
                "  Eval — NDCG@10: %.5f, Precision@10: %.5f, Recall@10: %.5f",
                metrics.ndcg,
                metrics.precision,
                metrics.recall,
            )

            # Save checkpoint.
            checkpoint_path = run_directory / "checkpoints" / f"epoch_{epoch + 1:03d}.pt"
            model.save_checkpoint(str(checkpoint_path))

            # Track best model.
            if metrics.ndcg > best_ndcg:
                best_ndcg = metrics.ndcg
                best_epoch = epoch + 1
                best_path = run_directory / "checkpoints" / "best.pt"
                model.save_checkpoint(str(best_path))
                logger.info(
                    "  New best model at epoch %d (NDCG@10: %.5f)",
                    best_epoch,
                    best_ndcg,
                )

        epoch_records.append(epoch_record)

        # ── Measure epoch duration and detect thermal throttling ────
        epoch_duration = time.time() - epoch_start_time
        epoch_durations.append(epoch_duration)
        logger.info("  Epoch duration: %.1f seconds", epoch_duration)

        # Save full training state after every epoch so we can resume
        # from the latest point if throttling triggers a cooldown.
        latest_checkpoint_path = run_directory / "checkpoints" / "latest.pt"
        _save_training_state(
            path=latest_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_ndcg=best_ndcg,
            best_epoch=best_epoch,
            epoch_records=epoch_records,
        )

        # Thermal throttle detection: if the current epoch took more than
        # _THROTTLE_FACTOR × the median of previous epochs, the GPU has
        # likely throttled.  Pause for _COOLDOWN_SECONDS and reload
        # the model state to let the hardware cool down.
        if len(epoch_durations) >= 3:
            median_duration = float(np.median(epoch_durations[:-1]))
            if epoch_duration > _THROTTLE_FACTOR * median_duration:
                logger.warning(
                    "Thermal throttle detected! Epoch took %.1fs "
                    "(median: %.1fs, threshold: %.1fs). "
                    "Pausing for %d seconds to cool down...",
                    epoch_duration,
                    median_duration,
                    _THROTTLE_FACTOR * median_duration,
                    _COOLDOWN_SECONDS,
                )

                # The epoch's work is already saved in latest.pt above.
                # Sleep to let the GPU cool down.
                time.sleep(_COOLDOWN_SECONDS)

                # Reload model + optimizer + scheduler state from the
                # checkpoint to ensure clean state on the cooled device.
                logger.info(
                    "Cooldown complete.  Reloading state from %s",
                    latest_checkpoint_path,
                )
                _load_training_state(
                    latest_checkpoint_path, model, optimizer, scheduler,
                )

                # Remove the anomalous duration so it doesn't poison the
                # rolling median for future detection.
                epoch_durations.pop()
                logger.info("Resuming training from epoch %d.", epoch + 2)

    total_training_time = time.time() - training_start_time
    logger.info(
        "Training complete in %.1f seconds. Best NDCG@10: %.5f at epoch %d",
        total_training_time,
        best_ndcg,
        best_epoch,
    )

    # ── Final evaluation with the best checkpoint ────────────────────
    logger.info("Running final evaluation with best checkpoint...")
    best_checkpoint_path = run_directory / "checkpoints" / "best.pt"
    if best_checkpoint_path.exists():
        model.load_checkpoint(str(best_checkpoint_path))
    model.eval_mode()
    model.faiss_index = None
    model.build_faiss_index()

    # Time the prediction step for performance reporting.
    prediction_start_time = time.time()
    final_offline_report = evaluate_offline(
        model=model,
        train_ratings=train_ratings,
        test_ratings=validation_ratings,
        users=users_dataframe,
        movies=movies_dataframe,
        ks=evaluation_ks,
        threshold=model.positive_threshold,
        mode="all",
    )
    total_prediction_time = time.time() - prediction_start_time
    number_of_users = int(users_dataframe["UserID"].nunique())
    time_per_user = total_prediction_time / max(number_of_users, 1)

    logger.info("=== Final Offline Evaluation ===")
    for k_value in sorted(final_offline_report.by_k):
        metrics_at_k = final_offline_report.by_k[k_value]
        logger.info(
            "  k=%d: NDCG=%.5f, Precision=%.5f, Recall=%.5f, MRR=%.5f, MAP=%.5f",
            k_value,
            metrics_at_k.ndcg,
            metrics_at_k.precision,
            metrics_at_k.recall,
            metrics_at_k.mrr,
            metrics_at_k.map,
        )

    return {
        "epochs": epoch_records,
        "best_epoch": best_epoch,
        "best_ndcg_at_10": best_ndcg,
        "total_training_time_seconds": total_training_time,
        "total_prediction_time_seconds": total_prediction_time,
        "time_per_user_seconds": time_per_user,
        "number_of_users": number_of_users,
        "final_offline_report": {
            k_value: {
                "ndcg": metrics_at_k.ndcg,
                "precision": metrics_at_k.precision,
                "recall": metrics_at_k.recall,
                "mrr": metrics_at_k.mrr,
                "map": metrics_at_k.map,
            }
            for k_value, metrics_at_k in final_offline_report.by_k.items()
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────


def generate_report(
    report_data: Dict,
    config: Dict,
    report_path: Path,
) -> None:
    """Write a Markdown post-training report.

    Parameters
    ----------
    report_data : dict
        Return value of ``train()``.
    config : dict
        Model and training hyperparameters.
    report_path : Path
        Output file path.
    """
    lines: List[str] = []
    lines.append("# Two-Tower Transformer — Training Report\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n")

    # ── Configuration ────────────────────────────────────────────────
    lines.append("## Configuration\n")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    for key, value in sorted(config.items()):
        lines.append(f"| {key} | {value} |")
    lines.append("")

    # ── Training history ─────────────────────────────────────────────
    lines.append("## Training History\n")
    lines.append("| Epoch | Loss | LR | NDCG@10 | Precision@10 | Recall@10 |")
    lines.append("|-------|------|----|---------|-------------|-----------|")
    for record in report_data["epochs"]:
        metrics = record.get("metrics")
        if metrics:
            lines.append(
                f"| {record['epoch']} "
                f"| {record['loss']:.6f} "
                f"| {record['learning_rate']:.2e} "
                f"| {metrics['ndcg_at_10']:.5f} "
                f"| {metrics['precision_at_10']:.5f} "
                f"| {metrics['recall_at_10']:.5f} |"
            )
        else:
            lines.append(
                f"| {record['epoch']} "
                f"| {record['loss']:.6f} "
                f"| {record['learning_rate']:.2e} "
                f"| — | — | — |"
            )
    lines.append("")

    # ── Best model ───────────────────────────────────────────────────
    lines.append("## Best Model\n")
    lines.append(f"- **Epoch:** {report_data['best_epoch']}")
    lines.append(f"- **NDCG@10:** {report_data['best_ndcg_at_10']:.5f}")
    lines.append("")

    # ── Final evaluation ─────────────────────────────────────────────
    lines.append("## Final Evaluation (Offline)\n")
    lines.append("| K | NDCG | Precision | Recall | MRR | MAP |")
    lines.append("|---|------|-----------|--------|-----|-----|")
    for k_value in sorted(report_data["final_offline_report"]):
        metrics = report_data["final_offline_report"][k_value]
        lines.append(
            f"| {k_value} "
            f"| {metrics['ndcg']:.5f} "
            f"| {metrics['precision']:.5f} "
            f"| {metrics['recall']:.5f} "
            f"| {metrics['mrr']:.5f} "
            f"| {metrics['map']:.5f} |"
        )
    lines.append("")

    # ── Performance ──────────────────────────────────────────────────
    lines.append("## Performance\n")
    lines.append(
        f"- **Total training time:** "
        f"{report_data['total_training_time_seconds']:.1f} seconds"
    )
    lines.append(
        f"- **Total prediction time:** "
        f"{report_data['total_prediction_time_seconds']:.1f} seconds "
        f"({report_data['number_of_users']} users)"
    )
    lines.append(
        f"- **Time per user:** "
        f"{report_data['time_per_user_seconds'] * 1000:.2f} ms"
    )
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to %s", report_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Two-Tower Transformer recommender.",
    )

    # ── Architecture ─────────────────────────────────────────────────
    parser.add_argument(
        "--projection-dimension", type=int, default=128,
        help="Output dimension for both towers (default: 128).",
    )
    parser.add_argument(
        "--item-hidden-dimension", type=int, default=256,
        help="Hidden dimension in the item tower MLP (default: 256).",
    )
    parser.add_argument(
        "--number-of-transformer-layers", type=int, default=2,
        help="Number of self-attention layers in the user tower (default: 2).",
    )
    parser.add_argument(
        "--number-of-attention-heads", type=int, default=4,
        help="Number of attention heads (default: 4).",
    )
    parser.add_argument(
        "--feedforward-dimension", type=int, default=256,
        help="FFN intermediate dimension in transformer layers (default: 256).",
    )
    parser.add_argument(
        "--maximum-history-length", type=int, default=64,
        help="Maximum number of history items per user (default: 64).",
    )
    parser.add_argument(
        "--dropout-rate", type=float, default=0.1,
        help="Dropout rate (default: 0.1).",
    )

    # ── Training ─────────────────────────────────────────────────────
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3,
        help="Peak learning rate for AdamW (default: 1e-3).",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-5,
        help="AdamW weight decay (default: 1e-5).",
    )
    parser.add_argument(
        "--number-of-epochs", type=int, default=30,
        help="Total training epochs (default: 30).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=512,
        help="Training batch size (default: 512).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="Temperature τ for InfoNCE loss (default: 0.1).",
    )
    parser.add_argument(
        "--positive-threshold", type=float, default=4.0,
        help="Rating threshold for positive interactions (default: 4.0).",
    )

    # ── Evaluation ───────────────────────────────────────────────────
    parser.add_argument(
        "--evaluation-interval", type=int, default=5,
        help="Evaluate every N epochs (default: 5).",
    )
    parser.add_argument(
        "--ks", type=str, default="10,20",
        help="Comma-separated K values for the final offline evaluation "
             "(default: '10,20').",
    )

    # ── Infrastructure ───────────────────────────────────────────────
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device (default: auto-detect).",
    )
    parser.add_argument(
        "--random-seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Custom run directory name (default: auto-generated timestamp).",
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Path to a checkpoint to resume training from.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Reproducibility ──────────────────────────────────────────────
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    # ── Run directory ────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "two_tower_transformer"
    run_name = args.run_name or f"{model_name}_{timestamp}"
    run_directory = Path("runs") / run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "checkpoints").mkdir(exist_ok=True)

    # ── Dual logging (console + file) ────────────────────────────────
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_format = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(run_directory / "training.log")
    file_handler.setFormatter(log_format)
    root_logger.addHandler(file_handler)

    logger.info("Run directory: %s", run_directory)

    # ── Load raw embeddings ──────────────────────────────────────────
    embeddings_path = Path("data/datasets/embeddings.npz")
    logger.info("Loading raw embeddings from %s", embeddings_path)
    raw_embeddings = np.load(str(embeddings_path))["concat"]
    logger.info(
        "Embeddings loaded: shape=%s, dtype=%s",
        raw_embeddings.shape,
        raw_embeddings.dtype,
    )

    # ── Build model ──────────────────────────────────────────────────
    model = TwoTowerTransformerRecommender(
        projection_dimension=args.projection_dimension,
        item_hidden_dimension=args.item_hidden_dimension,
        number_of_transformer_layers=args.number_of_transformer_layers,
        number_of_attention_heads=args.number_of_attention_heads,
        feedforward_dimension=args.feedforward_dimension,
        maximum_history_length=args.maximum_history_length,
        dropout_rate=args.dropout_rate,
        positive_threshold=args.positive_threshold,
        device=args.device,
    )

    model.prepare(
        movies_enriched=movies_enriched,
        raw_embeddings=raw_embeddings,
        train_ratings=user_based_temporal_train,
    )

    # ── Build dataset and dataloader ─────────────────────────────────
    logger.info("Building training dataset...")
    dataset = InteractionDataset(
        ratings=user_based_temporal_train,
        users_dataframe=users,
        raw_embeddings=raw_embeddings,
        movie_id_to_index=model.movie_id_to_index,
        positive_threshold=args.positive_threshold,
        maximum_history_length=args.maximum_history_length,
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # In-memory data, no I/O bottleneck.
        pin_memory=True,
        collate_fn=collate_interaction_batch,
        drop_last=True,  # Required for consistent in-batch negatives.
    )
    logger.info(
        "DataLoader: %d batches of size %d",
        len(train_dataloader),
        args.batch_size,
    )

    # ── Train ────────────────────────────────────────────────────────
    evaluation_ks = [
        int(k.strip()) for k in args.ks.split(",") if k.strip()
    ]

    config = {
        "projection_dimension": args.projection_dimension,
        "item_hidden_dimension": args.item_hidden_dimension,
        "number_of_transformer_layers": args.number_of_transformer_layers,
        "number_of_attention_heads": args.number_of_attention_heads,
        "feedforward_dimension": args.feedforward_dimension,
        "maximum_history_length": args.maximum_history_length,
        "dropout_rate": args.dropout_rate,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "number_of_epochs": args.number_of_epochs,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "positive_threshold": args.positive_threshold,
        "evaluation_interval": args.evaluation_interval,
        "evaluation_ks": evaluation_ks,
        "device": str(model.device),
        "random_seed": args.random_seed,
        "dataset_size": len(dataset),
        "number_of_movies": len(model.movie_id_to_index),
        "number_of_users": int(users["UserID"].nunique()),
    }
    logger.info("Configuration: %s", json.dumps(config, indent=2))

    report_data = train(
        model=model,
        train_dataloader=train_dataloader,
        validation_ratings=user_based_temporal_val,
        users_dataframe=users,
        movies_dataframe=movies_enriched,
        number_of_epochs=args.number_of_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        evaluation_interval=args.evaluation_interval,
        run_directory=run_directory,
        evaluation_ks=evaluation_ks,
        train_ratings=user_based_temporal_train,
        resume_from=args.resume_from,
    )

    # ── Generate report ──────────────────────────────────────────────
    generate_report(
        report_data=report_data,
        config=config,
        report_path=run_directory / "report.md",
    )

    logger.info("Done. Run directory: %s", run_directory)


if __name__ == "__main__":
    main()
