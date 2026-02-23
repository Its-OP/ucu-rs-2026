"""Arm selection strategies for multi-armed bandit model selection.

Implements the Strategy pattern: the abstract base class ``ArmSelectionStrategy``
defines the interface for arm selection policies, and concrete implementations
can be swapped in without modifying the bandit model selector or the simulation
loop.

Currently available strategies:

- **Epsilon-Greedy**: explores uniformly with probability epsilon, otherwise
  exploits the arm with the highest empirical mean reward.
- **Thompson Sampling (Beta-Bernoulli)**: models each arm's success probability
  with a Beta posterior, samples from each posterior, and picks the arm with the
  highest sample.  Continuous rewards in [0, 1] are binarized at a configurable
  threshold.

The design supports future extension with UCB, LinUCB, etc. by adding new
subclasses — no existing code needs to change.
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


class ThompsonSamplingStrategy(ArmSelectionStrategy):
    """Thompson Sampling arm selection with a Beta-Bernoulli reward model.

    Each arm ``a`` maintains a Beta posterior over its success probability:

        p_a ~ Beta(alpha_a, beta_a)

    At each time step the strategy:

    1. **Samples** theta_a ~ Beta(alpha_a, beta_a) for every arm a.
    2. **Selects** A_t = argmax_a theta_a   (ties broken by the sample values).
    3. **Observes** a continuous reward R_t in [0, 1] (e.g. NDCG@K).
    4. **Binarizes** the reward:
           success = 1  if R_t > reward_threshold
           success = 0  otherwise
    5. **Updates** the posterior:
           alpha_a <- alpha_a + success
           beta_a  <- beta_a  + (1 - success)

    The default ``reward_threshold = 0.0`` treats *any* positive NDCG as a
    success, which is appropriate when many users receive NDCG = 0 (cold-start
    or irrelevant recommendations).

    Parameters
    ----------
    reward_threshold : float
        Continuous reward values strictly above this threshold are treated as
        successes (Bernoulli 1); at or below it as failures (Bernoulli 0).
        Must be in [0, 1].
    prior_alpha : float
        Initial alpha parameter of the Beta prior for every arm.  Must be > 0.
        Default 1.0 gives a uniform prior Beta(1, 1).
    prior_beta : float
        Initial beta parameter of the Beta prior for every arm.  Must be > 0.
        Default 1.0 gives a uniform prior Beta(1, 1).
    random_state : int
        Seed for the random number generator, ensuring reproducibility.
    """

    def __init__(
        self,
        reward_threshold: float = 0.0,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        random_state: int = 42,
    ) -> None:
        if not 0.0 <= reward_threshold <= 1.0:
            raise ValueError(
                f"reward_threshold must be in [0, 1], got {reward_threshold}"
            )
        if prior_alpha <= 0.0:
            raise ValueError(
                f"prior_alpha must be > 0, got {prior_alpha}"
            )
        if prior_beta <= 0.0:
            raise ValueError(
                f"prior_beta must be > 0, got {prior_beta}"
            )

        self.reward_threshold = float(reward_threshold)
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self.random_state = int(random_state)

        self._rng: np.random.Generator | None = None
        self._number_of_arms: int = 0
        # Beta posterior parameters per arm: _alphas[a], _betas[a]
        self._alphas: np.ndarray = np.array([])
        self._betas: np.ndarray = np.array([])
        # Standard arm statistics (pull count and reward sum) for reporting
        self._arm_statistics: list[ArmStatistics] = []

    def initialize(self, number_of_arms: int) -> None:
        """Reset internal state for a new simulation.

        Initializes each arm's posterior to Beta(prior_alpha, prior_beta)
        and re-seeds the random generator.
        """
        if number_of_arms < 1:
            raise ValueError(
                f"number_of_arms must be >= 1, got {number_of_arms}"
            )
        self._number_of_arms = number_of_arms
        self._rng = np.random.default_rng(self.random_state)

        # Beta posterior parameters: alpha_a and beta_a for each arm a
        self._alphas = np.full(number_of_arms, self.prior_alpha)
        self._betas = np.full(number_of_arms, self.prior_beta)

        self._arm_statistics = [
            ArmStatistics(arm_index=index) for index in range(number_of_arms)
        ]

    def select_arm(self) -> int:
        """Select an arm by Thompson Sampling from the Beta posteriors.

        For each arm a, sample:
            theta_a ~ Beta(alpha_a, beta_a)

        Then select:
            A_t = argmax_a theta_a

        Returns
        -------
        int
            Index of the selected arm.
        """
        if self._rng is None:
            raise RuntimeError(
                "Strategy not initialized. Call initialize() before select_arm()."
            )

        # Sample from each arm's Beta posterior
        # theta_a ~ Beta(alpha_a, beta_a) for all arms simultaneously
        sampled_values = self._rng.beta(self._alphas, self._betas)

        return int(np.argmax(sampled_values))

    def update(self, chosen_arm: int, reward: float) -> None:
        """Update the chosen arm's Beta posterior after observing a reward.

        Binarize the reward:
            success = 1  if reward > reward_threshold
            success = 0  otherwise

        Update the Beta posterior:
            alpha_a <- alpha_a + success
            beta_a  <- beta_a  + (1 - success)

        Also updates standard arm statistics (pull count, reward sum) for
        reporting purposes.
        """
        # Binarize the continuous reward at the threshold.
        # Strict inequality (>) so that a reward of exactly 0.0 with
        # reward_threshold=0.0 is correctly treated as a failure.
        success = 1.0 if reward > self.reward_threshold else 0.0

        # Bayesian update: Beta(alpha + success, beta + failure)
        self._alphas[chosen_arm] += success
        self._betas[chosen_arm] += 1.0 - success

        # Standard statistics for reporting
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
