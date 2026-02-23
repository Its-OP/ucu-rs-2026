"""Multi-armed bandit model selector that wraps pre-fitted child models.

Each child model is an "arm".  The bandit delegates per-user arm selection
to a pluggable :class:`ArmSelectionStrategy` and returns the chosen arm's
recommendations.

This class implements :class:`RecommenderModel` so it integrates with
the existing evaluation framework (``src/eval/eval.evaluate``).

Important
---------
``predict()`` does **not** call ``strategy.update()`` — it uses the
strategy's current (possibly learned) state to make selections.  To run the
full learning loop, use :func:`simulation.run_bandit_simulation` instead.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from src.models.base import Rating, RecommenderModel
from src.models.bandit.strategy import ArmSelectionStrategy

logger = logging.getLogger(__name__)


class BanditModelSelector(RecommenderModel):
    """Meta-recommender that selects among pre-fitted child models per user.

    Parameters
    ----------
    arms : list[RecommenderModel]
        Pre-fitted child models.  Must not be empty.
    arm_names : list[str]
        Human-readable names for each arm (same length as *arms*).
    strategy : ArmSelectionStrategy
        Pluggable arm selection policy.  The strategy should already be
        initialized (e.g., via a prior simulation run) if you want to
        use a learned policy.  If it has not been initialized, the
        constructor will call ``initialize(len(arms))``.
    """

    def __init__(
        self,
        arms: list[RecommenderModel],
        arm_names: list[str],
        strategy: ArmSelectionStrategy,
    ) -> None:
        if len(arms) != len(arm_names):
            raise ValueError(
                f"Number of arms ({len(arms)}) must match number of "
                f"arm names ({len(arm_names)})"
            )
        if not arms:
            raise ValueError("Must provide at least one arm")

        self.arms = arms
        self.arm_names = arm_names
        self.strategy = strategy

    def predict(
        self,
        users: pd.DataFrame,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        k: int = 10,
    ) -> Dict[int, List[Rating]]:
        """Produce top-K recommendations using the bandit's current policy.

        Collects predictions from every arm, then for each user asks the
        strategy which arm's predictions to serve.  The strategy is
        **not updated** during this call — it uses its frozen state.

        Parameters
        ----------
        users : pd.DataFrame
            User side-information (UserID, Gender, Age, Occupation, Zip-code).
        ratings : pd.DataFrame
            Observed interactions (used by child models to filter seen items).
        movies : pd.DataFrame
            Movie side-information.
        k : int
            Number of recommendations per user.

        Returns
        -------
        dict[int, list[Rating]]
            Mapping from UserID to a list of :class:`Rating` objects sorted
            by score descending (length up to *k*).
        """
        # Collect predictions from every arm (one call per arm, not per user).
        all_arm_predictions: list[Dict[int, List[Rating]]] = [
            arm.predict(users, ratings, movies, k=k) for arm in self.arms
        ]

        combined_predictions: Dict[int, List[Rating]] = {}
        for user_id in users["UserID"].astype(int).values:
            user_id = int(user_id)
            chosen_arm_index = self.strategy.select_arm()
            combined_predictions[user_id] = all_arm_predictions[
                chosen_arm_index
            ].get(user_id, [])

        return combined_predictions
