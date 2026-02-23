"""Offline replay simulation for multi-armed bandit model selection.

Simulates an online bandit by processing users sequentially from the
evaluation split.  For each user the strategy selects an arm (model),
the chosen arm's pre-computed recommendations are scored against ground
truth via per-user NDCG@K, and the strategy is updated with the reward.

All child models are pre-fitted and frozen before the simulation begins;
only the bandit strategy's state evolves during the replay loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from src.eval.metrics.ndcg import ndcg_at_k
from src.models.base import Rating, RecommenderModel
from src.models.bandit.strategy import ArmSelectionStrategy, ArmStatistics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for the simulation report
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class UserDecisionRecord:
    """Record of a single bandit decision for one user.

    Attributes
    ----------
    user_id : int
        The user that received a recommendation.
    chosen_arm_index : int
        Index of the arm (model) selected by the strategy.
    chosen_arm_name : str
        Human-readable name of the selected arm.
    reward : float
        Per-user NDCG@K computed from the chosen arm's recommendations.
    recommendation_count : int
        Number of items returned by the chosen arm for this user.
    """

    user_id: int
    chosen_arm_index: int
    chosen_arm_name: str
    reward: float
    recommendation_count: int


@dataclass(frozen=True, kw_only=True)
class BanditSimulationReport:
    """Complete report from an offline bandit simulation run.

    Attributes
    ----------
    arm_names : list[str]
        Human-readable names of all arms in order.
    final_arm_statistics : list[ArmStatistics]
        Strategy-level statistics for each arm at the end of the simulation.
    user_decisions : list[UserDecisionRecord]
        Per-user decision log (one entry per processed user).
    mean_reward : float
        Mean NDCG@K reward across all processed users.
    per_arm_mean_reward : dict[str, float]
        Mean NDCG@K grouped by arm name.
    per_arm_selection_count : dict[str, int]
        Number of times each arm was selected.
    per_arm_selection_fraction : dict[str, float]
        Fraction of total selections for each arm.
    total_users_processed : int
        Number of users for whom a recommendation was served.
    users_skipped_no_ground_truth : int
        Number of users skipped because they had no relevant items in
        the evaluation split (rating >= relevance_threshold).
    """

    arm_names: list[str]
    final_arm_statistics: list[ArmStatistics]
    user_decisions: list[UserDecisionRecord]

    mean_reward: float
    per_arm_mean_reward: Dict[str, float]
    per_arm_selection_count: Dict[str, int]
    per_arm_selection_fraction: Dict[str, float]

    total_users_processed: int
    users_skipped_no_ground_truth: int


# ---------------------------------------------------------------------------
# Report compilation helper
# ---------------------------------------------------------------------------

def _compile_report(
    arm_names: list[str],
    final_arm_statistics: list[ArmStatistics],
    user_decisions: list[UserDecisionRecord],
    users_skipped_no_ground_truth: int,
) -> BanditSimulationReport:
    """Aggregate per-user decision records into a simulation report."""
    total_processed = len(user_decisions)

    # Per-arm reward accumulators
    arm_reward_sums: Dict[str, float] = {name: 0.0 for name in arm_names}
    arm_counts: Dict[str, int] = {name: 0 for name in arm_names}

    for record in user_decisions:
        arm_reward_sums[record.chosen_arm_name] += record.reward
        arm_counts[record.chosen_arm_name] += 1

    per_arm_mean_reward = {
        name: (arm_reward_sums[name] / arm_counts[name] if arm_counts[name] > 0 else 0.0)
        for name in arm_names
    }
    per_arm_selection_fraction = {
        name: (arm_counts[name] / total_processed if total_processed > 0 else 0.0)
        for name in arm_names
    }

    overall_mean = float(
        np.mean([record.reward for record in user_decisions])
    ) if user_decisions else 0.0

    return BanditSimulationReport(
        arm_names=arm_names,
        final_arm_statistics=final_arm_statistics,
        user_decisions=user_decisions,
        mean_reward=overall_mean,
        per_arm_mean_reward=per_arm_mean_reward,
        per_arm_selection_count=arm_counts,
        per_arm_selection_fraction=per_arm_selection_fraction,
        total_users_processed=total_processed,
        users_skipped_no_ground_truth=users_skipped_no_ground_truth,
    )


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------

def run_bandit_simulation(
    arms: list[RecommenderModel],
    arm_names: list[str],
    strategy: ArmSelectionStrategy,
    train_ratings: pd.DataFrame,
    evaluation_ratings: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 4.0,
    process_order: str = "temporal",
    random_state: int = 42,
) -> BanditSimulationReport:
    """Run an offline replay bandit simulation.

    Simulates online learning by processing users from the evaluation split
    one at a time.  At each step the strategy picks an arm, the arm's
    pre-computed recommendations are scored via per-user NDCG@K, and the
    strategy learns from the observed reward.

    Parameters
    ----------
    arms : list[RecommenderModel]
        Pre-fitted (frozen) child models.  Each model is one arm.
    arm_names : list[str]
        Human-readable name for each arm (same length as *arms*).
    strategy : ArmSelectionStrategy
        Pluggable arm selection policy (e.g., epsilon-greedy).
    train_ratings : pd.DataFrame
        Training interactions (passed to each arm's ``predict`` method
        so that already-seen items are filtered out).
    evaluation_ratings : pd.DataFrame
        Held-out interactions used as ground truth.
    users : pd.DataFrame
        User side-information table.
    movies : pd.DataFrame
        Movie side-information table.
    k : int
        Top-K cut-off for recommendations and NDCG computation.
    relevance_threshold : float
        Minimum rating to consider an item relevant.  Users with no
        relevant items in the evaluation split are skipped (matching the
        existing evaluator behaviour).
    process_order : str
        ``"temporal"`` processes users in order of their earliest
        timestamp in the evaluation split.  ``"shuffled"`` randomises
        the order for ablation studies.
    random_state : int
        Seed used when ``process_order="shuffled"``.

    Returns
    -------
    BanditSimulationReport
        Full simulation report including per-user decisions, per-arm
        statistics, and aggregated metrics.
    """
    if len(arms) != len(arm_names):
        raise ValueError(
            f"len(arms)={len(arms)} must equal len(arm_names)={len(arm_names)}"
        )
    if not arms:
        raise ValueError("Must provide at least one arm")

    # ------------------------------------------------------------------
    # Phase 1: Pre-compute predictions from all frozen arms.
    # Each arm.predict() returns dict[int, list[Rating]] for ALL users.
    # Calling once per arm is far more efficient than calling per-user.
    # ------------------------------------------------------------------
    logger.info("Pre-computing predictions for %d arms...", len(arms))
    all_arm_predictions: list[Dict[int, List[Rating]]] = []
    for index, arm in enumerate(arms):
        logger.info("  Arm %d (%s): generating predictions...", index, arm_names[index])
        predictions = arm.predict(users, train_ratings, movies, k=k)
        all_arm_predictions.append(predictions)
        logger.info("  Arm %d (%s): %d users predicted", index, arm_names[index], len(predictions))

    # ------------------------------------------------------------------
    # Phase 2: Build ground-truth lookup from the evaluation split.
    # ground_truth[user_id] = {movie_id: rating, ...}
    # ------------------------------------------------------------------
    ground_truth: Dict[int, Dict[int, float]] = {
        int(user_id): dict(zip(group["MovieID"].astype(int), group["Rating"].astype(float)))
        for user_id, group in evaluation_ratings.groupby("UserID")
    }

    # ------------------------------------------------------------------
    # Phase 3: Determine user processing order.
    # ------------------------------------------------------------------
    earliest_timestamps = evaluation_ratings.groupby("UserID")["Timestamp"].min()

    if process_order == "temporal":
        ordered_user_ids = earliest_timestamps.sort_values().index.tolist()
    elif process_order == "shuffled":
        rng = np.random.default_rng(random_state)
        ordered_user_ids = earliest_timestamps.index.tolist()
        rng.shuffle(ordered_user_ids)
    else:
        raise ValueError(
            f"Unknown process_order: {process_order!r}.  "
            "Must be 'temporal' or 'shuffled'."
        )

    logger.info(
        "Processing %d users in %s order", len(ordered_user_ids), process_order
    )

    # ------------------------------------------------------------------
    # Phase 4: Initialize the strategy.
    # ------------------------------------------------------------------
    strategy.initialize(number_of_arms=len(arms))

    # ------------------------------------------------------------------
    # Phase 5: Sequential simulation loop.
    # ------------------------------------------------------------------
    user_decisions: list[UserDecisionRecord] = []
    skipped_count = 0

    for user_id in ordered_user_ids:
        user_id = int(user_id)
        true_ratings = ground_truth.get(user_id, {})

        # Skip users with no relevant items — matches existing evaluator
        # behaviour in src/eval/eval.py and src/eval/offline_ranking.py.
        if not any(rating >= relevance_threshold for rating in true_ratings.values()):
            skipped_count += 1
            continue

        # Strategy picks an arm
        chosen_arm_index = strategy.select_arm()

        # Retrieve the chosen arm's pre-computed recommendations
        chosen_recommendations = all_arm_predictions[chosen_arm_index].get(
            user_id, []
        )
        ranked_item_ids = np.array(
            [rating.movie_id for rating in chosen_recommendations], dtype=int
        )

        # Compute per-user NDCG@K as the reward signal.
        # NDCG@K = DCG@K / IDCG@K, naturally bounded in [0, 1].
        reward = ndcg_at_k(ranked_item_ids, true_ratings, k=k)

        # Update the strategy with the observed (arm, reward) pair
        strategy.update(chosen_arm=chosen_arm_index, reward=reward)

        # Record this decision for the report
        user_decisions.append(
            UserDecisionRecord(
                user_id=user_id,
                chosen_arm_index=chosen_arm_index,
                chosen_arm_name=arm_names[chosen_arm_index],
                reward=reward,
                recommendation_count=len(chosen_recommendations),
            )
        )

    if skipped_count > 0:
        logger.warning(
            "Skipped %d/%d users with no relevant items (threshold=%.2f)",
            skipped_count,
            len(ordered_user_ids),
            relevance_threshold,
        )

    # ------------------------------------------------------------------
    # Phase 6: Compile the simulation report.
    # ------------------------------------------------------------------
    return _compile_report(
        arm_names=arm_names,
        final_arm_statistics=strategy.get_arm_statistics(),
        user_decisions=user_decisions,
        users_skipped_no_ground_truth=skipped_count,
    )
