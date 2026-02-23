"""Arm selection strategies for multi-armed bandit model selection.

Implements the Strategy pattern: the abstract base class ``ArmSelectionStrategy``
defines the interface for arm selection policies, and concrete implementations
(starting with epsilon-greedy) can be swapped in without modifying the bandit
model selector or the simulation loop.

The design supports future extension with UCB, Thompson Sampling, LinUCB, etc.
by adding new subclasses — no existing code needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class ArmStatistics:
    """Running statistics for a single bandit arm.

    Tracks how many times the arm has been pulled and the cumulative
    reward sum, from which the empirical mean reward is derived.

    Empirical mean reward:
        Q(a) = (1 / N(a)) * sum_{t : A_t = a} R_t

    where N(a) is the pull count and R_t is the reward at time t.
    """

    arm_index: int
    pull_count: int = 0
    reward_sum: float = 0.0

    @property
    def mean_reward(self) -> float:
        """Empirical mean reward: Q(a) = reward_sum / pull_count.

        Returns 0.0 if the arm has never been pulled.
        """
        if self.pull_count == 0:
            return 0.0
        return self.reward_sum / self.pull_count


class ArmSelectionStrategy(ABC):
    """Abstract base class for multi-armed bandit arm selection policies.

    Each strategy maintains its own internal state (arm statistics, random
    state) and exposes three operations:

    - ``initialize``: reset internal state for a new simulation
    - ``select_arm``: choose which arm to pull given current state
    - ``update``: incorporate a new reward observation

    The ``initialize`` method is separated from ``__init__`` so that a
    strategy object can be reused across multiple simulations without
    re-constructing it.
    """

    @abstractmethod
    def initialize(self, number_of_arms: int) -> None:
        """Reset internal state for a new simulation with the given number of arms.

        Parameters
        ----------
        number_of_arms : int
            Total number of arms available for selection.
        """
        ...

    @abstractmethod
    def select_arm(self) -> int:
        """Select an arm index to pull.

        Returns
        -------
        int
            Index of the selected arm (0-based).
        """
        ...

    @abstractmethod
    def update(self, chosen_arm: int, reward: float) -> None:
        """Update internal state after observing a reward for the chosen arm.

        Parameters
        ----------
        chosen_arm : int
            Index of the arm that was pulled.
        reward : float
            Observed reward signal (e.g., per-user NDCG@K in [0, 1]).
        """
        ...

    @abstractmethod
    def get_arm_statistics(self) -> list[ArmStatistics]:
        """Return current statistics for all arms.

        Returns
        -------
        list[ArmStatistics]
            One entry per arm, ordered by arm index.
        """
        ...


class EpsilonGreedyStrategy(ArmSelectionStrategy):
    """Epsilon-greedy arm selection policy.

    With probability epsilon, selects a uniformly random arm (exploration).
    With probability 1 - epsilon, selects the arm with the highest empirical
    mean reward (exploitation). Ties during exploitation are broken randomly.

    Selection rule:
        A_t = argmax_a Q_t(a)          with probability 1 - epsilon
        A_t = Uniform({0, ..., K-1})   with probability epsilon

    where Q_t(a) is the empirical mean reward for arm a at time t, and
    K is the total number of arms.

    Parameters
    ----------
    epsilon : float
        Exploration probability, must be in [0, 1].
    random_state : int
        Seed for the random number generator, ensuring reproducibility.
    """

    def __init__(self, epsilon: float = 0.1, random_state: int = 42) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(
                f"epsilon must be in [0, 1], got {epsilon}"
            )
        self.epsilon = float(epsilon)
        self.random_state = int(random_state)
        self._rng: np.random.Generator | None = None
        self._arm_statistics: list[ArmStatistics] = []
        self._number_of_arms: int = 0

    def initialize(self, number_of_arms: int) -> None:
        """Reset internal state for a new simulation.

        Creates fresh arm statistics and re-seeds the random generator.
        """
        if number_of_arms < 1:
            raise ValueError(
                f"number_of_arms must be >= 1, got {number_of_arms}"
            )
        self._number_of_arms = number_of_arms
        self._rng = np.random.default_rng(self.random_state)
        self._arm_statistics = [
            ArmStatistics(arm_index=index) for index in range(number_of_arms)
        ]

    def select_arm(self) -> int:
        """Select an arm using the epsilon-greedy rule.

        Epsilon-greedy selection:
            A_t = argmax_a Q_t(a)          with probability 1 - epsilon
            A_t = Uniform({0, ..., K-1})   with probability epsilon

        When exploiting (1 - epsilon branch), if multiple arms share the
        highest mean reward, one is chosen uniformly at random among them.

        Returns
        -------
        int
            Index of the selected arm.
        """
        if self._rng is None:
            raise RuntimeError(
                "Strategy not initialized. Call initialize() before select_arm()."
            )

        # Exploration: pick a random arm
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self._number_of_arms))

        # Exploitation: pick the arm with the highest empirical mean reward.
        # Break ties randomly among arms sharing the maximum.
        mean_rewards = np.array(
            [arm.mean_reward for arm in self._arm_statistics]
        )
        maximum_reward = np.max(mean_rewards)
        best_arm_indices = np.flatnonzero(mean_rewards == maximum_reward)

        if len(best_arm_indices) == 1:
            return int(best_arm_indices[0])
        return int(self._rng.choice(best_arm_indices))

    def update(self, chosen_arm: int, reward: float) -> None:
        """Update the chosen arm's statistics with the observed reward.

        Increments the pull count and adds the reward to the cumulative sum:
            N(a) <- N(a) + 1
            S(a) <- S(a) + R_t
        so that Q(a) = S(a) / N(a).
        """
        self._arm_statistics[chosen_arm].pull_count += 1
        self._arm_statistics[chosen_arm].reward_sum += reward

    def get_arm_statistics(self) -> list[ArmStatistics]:
        """Return a copy of the current arm statistics."""
        return [
            ArmStatistics(
                arm_index=arm.arm_index,
                pull_count=arm.pull_count,
                reward_sum=arm.reward_sum,
            )
            for arm in self._arm_statistics
        ]
