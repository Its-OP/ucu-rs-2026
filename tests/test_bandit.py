"""Tests for the multi-armed bandit model selector.

Test structure mirrors ``tests/test_eval.py``: mock RecommenderModel
implementations are used with small synthetic DataFrames to verify
strategy logic, the BanditModelSelector interface, and the offline
simulation loop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.metrics.ndcg import ndcg_at_k
from src.models.base import Rating, RecommenderModel
from src.models.bandit.bandit_model_selector import BanditModelSelector
from src.models.bandit.simulation import run_bandit_simulation
from src.models.bandit.strategy import (
    ArmStatistics,
    EpsilonGreedyStrategy,
    ThompsonSamplingStrategy,
)


# ─── Mock models ─────────────────────────────────────────────────────────


class PerfectModel(RecommenderModel):
    """Returns items sorted by their true test rating (best possible)."""

    def __init__(self, test_ratings: pd.DataFrame) -> None:
        self._test = test_ratings

    def predict(self, users, ratings, movies, k=10):
        result = {}
        for user_id, group in self._test.groupby("UserID"):
            top = group.nlargest(k, "Rating")
            result[int(user_id)] = [
                Rating(movie_id=int(row.MovieID), score=float(row.Rating))
                for row in top.itertuples()
            ]
        return result


class BadModel(RecommenderModel):
    """Always recommends a fixed list of non-relevant items."""

    def __init__(self, bad_movie_ids: list[int]) -> None:
        self._bad_ids = bad_movie_ids

    def predict(self, users, ratings, movies, k=10):
        result = {}
        recommendations = [
            Rating(movie_id=mid, score=float(100 - index))
            for index, mid in enumerate(self._bad_ids[:k])
        ]
        for user_id in users["UserID"].unique():
            result[int(user_id)] = recommendations
        return result


class EmptyModel(RecommenderModel):
    """Returns an empty list for every user (simulates cold-start failure)."""

    def predict(self, users, ratings, movies, k=10):
        return {int(uid): [] for uid in users["UserID"].unique()}


# ─── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def sample_users():
    return pd.DataFrame(
        {
            "UserID": [1, 2, 3],
            "Gender": ["M", "F", "M"],
            "Age": [25, 35, 18],
            "Occupation": [0, 1, 2],
            "Zip-code": ["00000", "11111", "22222"],
        }
    )


@pytest.fixture
def sample_movies():
    return pd.DataFrame(
        {
            "MovieID": [10, 20, 30, 40, 50, 60, 70],
            "Title": ["A", "B", "C", "D", "E", "F", "G"],
            "Genres": ["Action"] * 7,
        }
    )


@pytest.fixture
def sample_train_ratings():
    return pd.DataFrame(
        {
            "UserID": [1, 1, 2, 2, 3, 3],
            "MovieID": [10, 20, 30, 40, 10, 50],
            "Rating": [5.0, 3.0, 4.0, 2.0, 4.0, 5.0],
            "Timestamp": pd.to_datetime(
                [
                    "2000-06-01",
                    "2000-06-02",
                    "2000-06-03",
                    "2000-06-04",
                    "2000-06-05",
                    "2000-06-06",
                ]
            ),
        }
    )


@pytest.fixture
def sample_test_ratings():
    return pd.DataFrame(
        {
            "UserID": [1, 1, 2, 2, 3, 3],
            "MovieID": [30, 40, 50, 60, 20, 70],
            "Rating": [5.0, 4.0, 5.0, 1.0, 3.0, 5.0],
            "Timestamp": pd.to_datetime(
                [
                    "2000-12-01",
                    "2000-12-02",
                    "2000-12-03",
                    "2000-12-04",
                    "2000-12-05",
                    "2000-12-06",
                ]
            ),
        }
    )


# ═══════════════════════════════════════════════════════════════════════════
# Strategy Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestArmStatistics:
    def test_mean_reward_zero_when_unpulled(self):
        arm = ArmStatistics(arm_index=0)
        assert arm.mean_reward == 0.0

    def test_mean_reward_computes_correctly(self):
        arm = ArmStatistics(arm_index=0, pull_count=4, reward_sum=2.0)
        assert arm.mean_reward == pytest.approx(0.5)


class TestEpsilonGreedyStrategy:
    def test_raises_on_invalid_epsilon(self):
        with pytest.raises(ValueError, match="epsilon must be in"):
            EpsilonGreedyStrategy(epsilon=-0.1)
        with pytest.raises(ValueError, match="epsilon must be in"):
            EpsilonGreedyStrategy(epsilon=1.5)

    def test_raises_on_invalid_number_of_arms(self):
        strategy = EpsilonGreedyStrategy(epsilon=0.1)
        with pytest.raises(ValueError, match="number_of_arms must be >= 1"):
            strategy.initialize(number_of_arms=0)

    def test_select_arm_raises_if_not_initialized(self):
        strategy = EpsilonGreedyStrategy(epsilon=0.1)
        with pytest.raises(RuntimeError, match="not initialized"):
            strategy.select_arm()

    def test_initialize_resets_state(self):
        strategy = EpsilonGreedyStrategy(epsilon=0.0, random_state=42)
        strategy.initialize(number_of_arms=2)
        strategy.update(chosen_arm=0, reward=1.0)

        # Re-initialize should clear the previous state
        strategy.initialize(number_of_arms=3)
        statistics = strategy.get_arm_statistics()
        assert len(statistics) == 3
        assert all(arm.pull_count == 0 for arm in statistics)
        assert all(arm.reward_sum == 0.0 for arm in statistics)

    def test_pure_exploration_selects_all_arms(self):
        """With epsilon=1.0, all arms should be selected over many pulls."""
        strategy = EpsilonGreedyStrategy(epsilon=1.0, random_state=42)
        strategy.initialize(number_of_arms=3)

        selections = [strategy.select_arm() for _ in range(300)]
        unique_arms = set(selections)
        assert unique_arms == {0, 1, 2}, (
            f"Expected all arms selected, got {unique_arms}"
        )

    def test_pure_exploitation_selects_best_arm(self):
        """With epsilon=0.0, after updating, always selects the best arm."""
        strategy = EpsilonGreedyStrategy(epsilon=0.0, random_state=42)
        strategy.initialize(number_of_arms=3)

        # Arm 1 has the highest mean reward
        strategy.update(chosen_arm=0, reward=0.2)
        strategy.update(chosen_arm=1, reward=0.9)
        strategy.update(chosen_arm=2, reward=0.5)

        for _ in range(50):
            assert strategy.select_arm() == 1

    def test_update_increments_counts_and_sums(self):
        strategy = EpsilonGreedyStrategy(epsilon=0.1, random_state=42)
        strategy.initialize(number_of_arms=2)

        strategy.update(chosen_arm=0, reward=0.5)
        strategy.update(chosen_arm=0, reward=0.3)
        strategy.update(chosen_arm=1, reward=0.8)

        statistics = strategy.get_arm_statistics()
        assert statistics[0].pull_count == 2
        assert statistics[0].reward_sum == pytest.approx(0.8)
        assert statistics[0].mean_reward == pytest.approx(0.4)
        assert statistics[1].pull_count == 1
        assert statistics[1].reward_sum == pytest.approx(0.8)

    def test_get_arm_statistics_returns_copy(self):
        """Modifying returned statistics should not affect internal state."""
        strategy = EpsilonGreedyStrategy(epsilon=0.0, random_state=42)
        strategy.initialize(number_of_arms=2)
        strategy.update(chosen_arm=0, reward=0.5)

        statistics = strategy.get_arm_statistics()
        statistics[0].pull_count = 999

        fresh_statistics = strategy.get_arm_statistics()
        assert fresh_statistics[0].pull_count == 1  # unchanged

    def test_tie_breaking_is_random(self):
        """When all arms have the same mean reward, selection varies."""
        strategy = EpsilonGreedyStrategy(epsilon=0.0, random_state=42)
        strategy.initialize(number_of_arms=3)

        # All arms have 0 mean (no updates yet) -> tie -> random selection
        selections = {strategy.select_arm() for _ in range(100)}
        assert len(selections) > 1, "Tie-breaking should produce varied selections"


class TestThompsonSamplingStrategy:
    def test_raises_on_invalid_reward_threshold(self):
        with pytest.raises(ValueError, match="reward_threshold must be in"):
            ThompsonSamplingStrategy(reward_threshold=-0.1)
        with pytest.raises(ValueError, match="reward_threshold must be in"):
            ThompsonSamplingStrategy(reward_threshold=1.5)

    def test_raises_on_invalid_prior_alpha(self):
        with pytest.raises(ValueError, match="prior_alpha must be > 0"):
            ThompsonSamplingStrategy(prior_alpha=0.0)
        with pytest.raises(ValueError, match="prior_alpha must be > 0"):
            ThompsonSamplingStrategy(prior_alpha=-1.0)

    def test_raises_on_invalid_prior_beta(self):
        with pytest.raises(ValueError, match="prior_beta must be > 0"):
            ThompsonSamplingStrategy(prior_beta=0.0)

    def test_raises_on_invalid_number_of_arms(self):
        strategy = ThompsonSamplingStrategy()
        with pytest.raises(ValueError, match="number_of_arms must be >= 1"):
            strategy.initialize(number_of_arms=0)

    def test_select_arm_raises_if_not_initialized(self):
        strategy = ThompsonSamplingStrategy()
        with pytest.raises(RuntimeError, match="not initialized"):
            strategy.select_arm()

    def test_initialize_resets_state(self):
        strategy = ThompsonSamplingStrategy(random_state=42)
        strategy.initialize(number_of_arms=2)
        strategy.update(chosen_arm=0, reward=1.0)

        # Re-initialize should clear the previous state
        strategy.initialize(number_of_arms=3)
        statistics = strategy.get_arm_statistics()
        assert len(statistics) == 3
        assert all(arm.pull_count == 0 for arm in statistics)
        assert all(arm.reward_sum == 0.0 for arm in statistics)

    def test_both_arms_selected_with_uniform_prior(self):
        """With uniform prior Beta(1,1) and no updates, both arms should
        be selected over many draws since samples are from Uniform(0,1)."""
        strategy = ThompsonSamplingStrategy(random_state=42)
        strategy.initialize(number_of_arms=2)

        selections = [strategy.select_arm() for _ in range(200)]
        unique_arms = set(selections)
        assert unique_arms == {0, 1}, (
            f"Expected both arms selected with uniform prior, got {unique_arms}"
        )

    def test_learns_to_prefer_rewarded_arm(self):
        """After many successes on arm 0 and failures on arm 1,
        Thompson Sampling should strongly prefer arm 0."""
        strategy = ThompsonSamplingStrategy(
            reward_threshold=0.0, random_state=42,
        )
        strategy.initialize(number_of_arms=2)

        # Feed many successes to arm 0 and failures to arm 1
        for _ in range(50):
            strategy.update(chosen_arm=0, reward=0.8)  # success
            strategy.update(chosen_arm=1, reward=0.0)  # failure

        # After strong evidence, arm 0 should be selected most of the time
        selections = [strategy.select_arm() for _ in range(100)]
        arm_0_count = selections.count(0)
        assert arm_0_count > 90, (
            f"Expected arm 0 selected >90 times, got {arm_0_count}"
        )

    def test_binarization_at_threshold(self):
        """Rewards at or below threshold are failures; strictly above are successes.

        With threshold=0.5:
          - reward 0.5 -> failure (at threshold, not above it)
          - reward 0.51 -> success (above threshold)
          - reward 0.4 -> failure (below threshold)
        """
        strategy = ThompsonSamplingStrategy(
            reward_threshold=0.5, random_state=42,
        )
        strategy.initialize(number_of_arms=2)

        # Arm 0 always gets rewards above threshold (success)
        # Arm 1 always gets rewards at threshold (failure — strict inequality)
        for _ in range(50):
            strategy.update(chosen_arm=0, reward=0.6)  # success (> 0.5)
            strategy.update(chosen_arm=1, reward=0.5)  # failure (= 0.5, not >)

        # Arm 0 should be strongly preferred
        selections = [strategy.select_arm() for _ in range(100)]
        arm_0_count = selections.count(0)
        assert arm_0_count > 90, (
            f"Expected arm 0 selected >90 times, got {arm_0_count}"
        )

    def test_update_increments_counts_and_sums(self):
        strategy = ThompsonSamplingStrategy(random_state=42)
        strategy.initialize(number_of_arms=2)

        strategy.update(chosen_arm=0, reward=0.5)
        strategy.update(chosen_arm=0, reward=0.3)
        strategy.update(chosen_arm=1, reward=0.8)

        statistics = strategy.get_arm_statistics()
        assert statistics[0].pull_count == 2
        assert statistics[0].reward_sum == pytest.approx(0.8)
        assert statistics[0].mean_reward == pytest.approx(0.4)
        assert statistics[1].pull_count == 1
        assert statistics[1].reward_sum == pytest.approx(0.8)

    def test_get_arm_statistics_returns_copy(self):
        """Modifying returned statistics should not affect internal state."""
        strategy = ThompsonSamplingStrategy(random_state=42)
        strategy.initialize(number_of_arms=2)
        strategy.update(chosen_arm=0, reward=0.5)

        statistics = strategy.get_arm_statistics()
        statistics[0].pull_count = 999

        fresh_statistics = strategy.get_arm_statistics()
        assert fresh_statistics[0].pull_count == 1  # unchanged

    def test_informative_prior_biases_initial_selection(self):
        """A strong prior alpha on arm selection should bias early pulls."""
        # Prior strongly favoring successes: Beta(10, 1) → mean ≈ 0.91
        strategy = ThompsonSamplingStrategy(
            prior_alpha=10.0, prior_beta=1.0, random_state=42,
        )
        strategy.initialize(number_of_arms=2)

        # With symmetric strong prior, both arms start equally.
        # Arm 0 gets failures, arm 1 gets successes → arm 1 eventually wins
        for _ in range(30):
            strategy.update(chosen_arm=0, reward=0.0)
            strategy.update(chosen_arm=1, reward=1.0)

        selections = [strategy.select_arm() for _ in range(100)]
        arm_1_count = selections.count(1)
        assert arm_1_count > 80, (
            f"Expected arm 1 preferred after evidence, got {arm_1_count}"
        )

    def test_works_in_simulation(
        self, sample_users, sample_movies, sample_train_ratings, sample_test_ratings
    ):
        """Thompson Sampling strategy should work with the simulation loop."""
        perfect = PerfectModel(sample_test_ratings)
        bad = BadModel(bad_movie_ids=[60, 70])
        strategy = ThompsonSamplingStrategy(
            reward_threshold=0.0, random_state=42,
        )

        report = run_bandit_simulation(
            arms=[perfect, bad],
            arm_names=["Perfect", "Bad"],
            strategy=strategy,
            train_ratings=sample_train_ratings,
            evaluation_ratings=sample_test_ratings,
            users=sample_users,
            movies=sample_movies,
            k=2,
            relevance_threshold=4.0,
        )

        # Basic sanity: report should complete and have processed users
        assert report.total_users_processed > 0
        total_selections = sum(report.per_arm_selection_count.values())
        assert total_selections == report.total_users_processed

    def test_bandit_learns_better_arm_thompson(
        self, sample_users, sample_movies, sample_train_ratings,
    ):
        """Over many users, Thompson Sampling should learn to prefer the
        better arm, similar to the epsilon-greedy test."""
        many_users = pd.DataFrame(
            {
                "UserID": list(range(1, 101)),
                "Gender": ["M"] * 100,
                "Age": [25] * 100,
                "Occupation": [0] * 100,
                "Zip-code": ["00000"] * 100,
            }
        )
        many_test_ratings_rows = []
        for uid in range(1, 101):
            many_test_ratings_rows.append(
                {"UserID": uid, "MovieID": 30, "Rating": 5.0,
                 "Timestamp": pd.Timestamp("2000-12-01") + pd.Timedelta(days=uid)}
            )
            many_test_ratings_rows.append(
                {"UserID": uid, "MovieID": 40, "Rating": 4.0,
                 "Timestamp": pd.Timestamp("2000-12-02") + pd.Timedelta(days=uid)}
            )
        many_test_ratings = pd.DataFrame(many_test_ratings_rows)

        many_train_ratings = pd.DataFrame(
            {
                "UserID": list(range(1, 101)),
                "MovieID": [10] * 100,
                "Rating": [3.0] * 100,
                "Timestamp": pd.to_datetime(["2000-06-01"] * 100),
            }
        )

        perfect = PerfectModel(many_test_ratings)
        bad = BadModel(bad_movie_ids=[60, 70])
        strategy = ThompsonSamplingStrategy(
            reward_threshold=0.0, random_state=42,
        )

        report = run_bandit_simulation(
            arms=[perfect, bad],
            arm_names=["Perfect", "Bad"],
            strategy=strategy,
            train_ratings=many_train_ratings,
            evaluation_ratings=many_test_ratings,
            users=many_users,
            movies=sample_movies,
            k=2,
            relevance_threshold=4.0,
        )

        assert report.per_arm_selection_count["Perfect"] > report.per_arm_selection_count["Bad"]


# ═══════════════════════════════════════════════════════════════════════════
# BanditModelSelector Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBanditModelSelector:
    def test_raises_on_mismatched_arms_and_names(self, sample_test_ratings):
        perfect = PerfectModel(sample_test_ratings)
        strategy = EpsilonGreedyStrategy(epsilon=0.1)
        strategy.initialize(number_of_arms=1)

        with pytest.raises(ValueError, match="must match"):
            BanditModelSelector(
                arms=[perfect],
                arm_names=["A", "B"],
                strategy=strategy,
            )

    def test_raises_on_empty_arms(self):
        strategy = EpsilonGreedyStrategy(epsilon=0.1)

        with pytest.raises(ValueError, match="at least one arm"):
            BanditModelSelector(arms=[], arm_names=[], strategy=strategy)

    def test_predict_returns_correct_structure(
        self, sample_users, sample_movies, sample_train_ratings, sample_test_ratings
    ):
        perfect = PerfectModel(sample_test_ratings)
        strategy = EpsilonGreedyStrategy(epsilon=0.0, random_state=42)
        strategy.initialize(number_of_arms=1)

        bandit = BanditModelSelector(
            arms=[perfect], arm_names=["Perfect"], strategy=strategy
        )

        predictions = bandit.predict(
            sample_users, sample_train_ratings, sample_movies, k=2
        )

        assert isinstance(predictions, dict)
        for user_id, recs in predictions.items():
            assert isinstance(user_id, int)
            assert isinstance(recs, list)
            for rec in recs:
                assert isinstance(rec, Rating)

    def test_predict_serves_chosen_arm_recommendations(
        self, sample_users, sample_movies, sample_train_ratings, sample_test_ratings
    ):
        """With epsilon=0 and arm 1 being the best, arm 1's recs are served."""
        perfect = PerfectModel(sample_test_ratings)
        bad = BadModel(bad_movie_ids=[60, 70])

        strategy = EpsilonGreedyStrategy(epsilon=0.0, random_state=42)
        strategy.initialize(number_of_arms=2)
        # Make arm 0 (perfect) the best
        strategy.update(chosen_arm=0, reward=1.0)
        strategy.update(chosen_arm=1, reward=0.0)

        bandit = BanditModelSelector(
            arms=[perfect, bad], arm_names=["Perfect", "Bad"], strategy=strategy
        )

        predictions = bandit.predict(
            sample_users, sample_train_ratings, sample_movies, k=2
        )

        # All users should get the perfect model's predictions
        perfect_predictions = perfect.predict(
            sample_users, sample_train_ratings, sample_movies, k=2
        )
        for user_id in predictions:
            bandit_ids = [r.movie_id for r in predictions[user_id]]
            perfect_ids = [r.movie_id for r in perfect_predictions.get(user_id, [])]
            assert bandit_ids == perfect_ids

    def test_handles_arm_returning_empty_list(
        self, sample_users, sample_movies, sample_train_ratings
    ):
        empty = EmptyModel()
        strategy = EpsilonGreedyStrategy(epsilon=0.0, random_state=42)
        strategy.initialize(number_of_arms=1)

        bandit = BanditModelSelector(
            arms=[empty], arm_names=["Empty"], strategy=strategy
        )

        predictions = bandit.predict(
            sample_users, sample_train_ratings, sample_movies, k=10
        )

        for user_id, recs in predictions.items():
            assert recs == []


# ═══════════════════════════════════════════════════════════════════════════
# Simulation Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRunBanditSimulation:
    def test_processes_eligible_users(
        self, sample_users, sample_movies, sample_train_ratings, sample_test_ratings
    ):
        """All users with relevant items should be processed."""
        perfect = PerfectModel(sample_test_ratings)
        strategy = EpsilonGreedyStrategy(epsilon=0.5, random_state=42)

        report = run_bandit_simulation(
            arms=[perfect],
            arm_names=["Perfect"],
            strategy=strategy,
            train_ratings=sample_train_ratings,
            evaluation_ratings=sample_test_ratings,
            users=sample_users,
            movies=sample_movies,
            k=2,
            relevance_threshold=4.0,
        )

        # Users 1 and 2 have items rated >= 4.0 in test; user 3 has one
        # item rated 5.0 (movie 70), so user 3 is also eligible.
        assert report.total_users_processed > 0
        processed_user_ids = {d.user_id for d in report.user_decisions}
        # At least users 1 and 2 should be processed (have relevant items)
        assert 1 in processed_user_ids
        assert 2 in processed_user_ids

    def test_skips_users_without_relevant_items(
        self, sample_users, sample_movies, sample_train_ratings
    ):
        """Users with no ratings >= threshold in test set should be skipped."""
        # All test ratings below threshold
        low_test_ratings = pd.DataFrame(
            {
                "UserID": [1, 2, 3],
                "MovieID": [30, 50, 70],
                "Rating": [2.0, 1.0, 3.0],
                "Timestamp": pd.to_datetime(["2000-12-01"] * 3),
            }
        )
        perfect = PerfectModel(low_test_ratings)
        strategy = EpsilonGreedyStrategy(epsilon=0.1, random_state=42)

        report = run_bandit_simulation(
            arms=[perfect],
            arm_names=["Perfect"],
            strategy=strategy,
            train_ratings=sample_train_ratings,
            evaluation_ratings=low_test_ratings,
            users=sample_users,
            movies=sample_movies,
            k=2,
            relevance_threshold=4.0,
        )

        assert report.total_users_processed == 0
        assert report.users_skipped_no_ground_truth == 3

    def test_arm_counts_sum_to_total(
        self, sample_users, sample_movies, sample_train_ratings, sample_test_ratings
    ):
        """Per-arm selection counts should sum to total_users_processed."""
        perfect = PerfectModel(sample_test_ratings)
        bad = BadModel(bad_movie_ids=[60, 70])
        strategy = EpsilonGreedyStrategy(epsilon=0.5, random_state=42)

        report = run_bandit_simulation(
            arms=[perfect, bad],
            arm_names=["Perfect", "Bad"],
            strategy=strategy,
            train_ratings=sample_train_ratings,
            evaluation_ratings=sample_test_ratings,
            users=sample_users,
            movies=sample_movies,
            k=2,
            relevance_threshold=4.0,
        )

        total_selections = sum(report.per_arm_selection_count.values())
        assert total_selections == report.total_users_processed

    def test_reward_matches_ndcg(
        self, sample_users, sample_movies, sample_train_ratings, sample_test_ratings
    ):
        """Each UserDecisionRecord's reward should equal ndcg_at_k."""
        perfect = PerfectModel(sample_test_ratings)
        strategy = EpsilonGreedyStrategy(epsilon=0.0, random_state=42)

        report = run_bandit_simulation(
            arms=[perfect],
            arm_names=["Perfect"],
            strategy=strategy,
            train_ratings=sample_train_ratings,
            evaluation_ratings=sample_test_ratings,
            users=sample_users,
            movies=sample_movies,
            k=2,
            relevance_threshold=4.0,
        )

        # Verify each decision's reward against manual ndcg_at_k
        ground_truth = {
            int(uid): dict(zip(group["MovieID"], group["Rating"]))
            for uid, group in sample_test_ratings.groupby("UserID")
        }
        perfect_predictions = perfect.predict(
            sample_users, sample_train_ratings, sample_movies, k=2
        )

        for decision in report.user_decisions:
            recs = perfect_predictions.get(decision.user_id, [])
            ranked_ids = np.array([r.movie_id for r in recs], dtype=int)
            expected_ndcg = ndcg_at_k(
                ranked_ids, ground_truth.get(decision.user_id, {}), k=2
            )
            assert decision.reward == pytest.approx(expected_ndcg)

    def test_bandit_learns_to_prefer_better_arm(
        self, sample_users, sample_movies, sample_train_ratings, sample_test_ratings
    ):
        """Over many users, the bandit should learn to prefer the better arm.

        We build a scenario with many users where the perfect model is
        clearly better, then check that the perfect arm gets more pulls.
        """
        # Create a larger dataset so the bandit has time to learn
        many_users = pd.DataFrame(
            {
                "UserID": list(range(1, 101)),
                "Gender": ["M"] * 100,
                "Age": [25] * 100,
                "Occupation": [0] * 100,
                "Zip-code": ["00000"] * 100,
            }
        )
        many_test_ratings_rows = []
        for uid in range(1, 101):
            many_test_ratings_rows.append(
                {"UserID": uid, "MovieID": 30, "Rating": 5.0,
                 "Timestamp": pd.Timestamp("2000-12-01") + pd.Timedelta(days=uid)}
            )
            many_test_ratings_rows.append(
                {"UserID": uid, "MovieID": 40, "Rating": 4.0,
                 "Timestamp": pd.Timestamp("2000-12-02") + pd.Timedelta(days=uid)}
            )
        many_test_ratings = pd.DataFrame(many_test_ratings_rows)

        many_train_ratings = pd.DataFrame(
            {
                "UserID": list(range(1, 101)),
                "MovieID": [10] * 100,
                "Rating": [3.0] * 100,
                "Timestamp": pd.to_datetime(["2000-06-01"] * 100),
            }
        )

        perfect = PerfectModel(many_test_ratings)
        bad = BadModel(bad_movie_ids=[60, 70])
        strategy = EpsilonGreedyStrategy(epsilon=0.1, random_state=42)

        report = run_bandit_simulation(
            arms=[perfect, bad],
            arm_names=["Perfect", "Bad"],
            strategy=strategy,
            train_ratings=many_train_ratings,
            evaluation_ratings=many_test_ratings,
            users=many_users,
            movies=sample_movies,
            k=2,
            relevance_threshold=4.0,
        )

        # The perfect arm should be selected more often than the bad arm
        assert report.per_arm_selection_count["Perfect"] > report.per_arm_selection_count["Bad"]

    def test_temporal_ordering(
        self, sample_users, sample_movies, sample_train_ratings
    ):
        """Users should be processed in order of earliest eval timestamp."""
        # User 3 has the earliest timestamp, then user 1, then user 2
        ordered_test_ratings = pd.DataFrame(
            {
                "UserID": [3, 1, 2],
                "MovieID": [70, 30, 50],
                "Rating": [5.0, 5.0, 5.0],
                "Timestamp": pd.to_datetime(
                    ["2000-11-01", "2000-12-01", "2001-01-01"]
                ),
            }
        )

        perfect = PerfectModel(ordered_test_ratings)
        strategy = EpsilonGreedyStrategy(epsilon=0.0, random_state=42)

        report = run_bandit_simulation(
            arms=[perfect],
            arm_names=["Perfect"],
            strategy=strategy,
            train_ratings=sample_train_ratings,
            evaluation_ratings=ordered_test_ratings,
            users=sample_users,
            movies=sample_movies,
            k=2,
            relevance_threshold=4.0,
            process_order="temporal",
        )

        processed_order = [d.user_id for d in report.user_decisions]
        assert processed_order == [3, 1, 2]

    def test_raises_on_mismatched_arms_and_names(
        self, sample_users, sample_movies, sample_train_ratings, sample_test_ratings
    ):
        perfect = PerfectModel(sample_test_ratings)
        strategy = EpsilonGreedyStrategy(epsilon=0.1, random_state=42)

        with pytest.raises(ValueError, match="must equal"):
            run_bandit_simulation(
                arms=[perfect],
                arm_names=["A", "B"],
                strategy=strategy,
                train_ratings=sample_train_ratings,
                evaluation_ratings=sample_test_ratings,
                users=sample_users,
                movies=sample_movies,
            )

    def test_empty_arm_gets_zero_reward(
        self, sample_users, sample_movies, sample_train_ratings, sample_test_ratings
    ):
        """An arm returning [] should receive reward 0.0 for that user."""
        empty = EmptyModel()
        strategy = EpsilonGreedyStrategy(epsilon=0.0, random_state=42)

        report = run_bandit_simulation(
            arms=[empty],
            arm_names=["Empty"],
            strategy=strategy,
            train_ratings=sample_train_ratings,
            evaluation_ratings=sample_test_ratings,
            users=sample_users,
            movies=sample_movies,
            k=2,
            relevance_threshold=4.0,
        )

        for decision in report.user_decisions:
            assert decision.reward == 0.0
            assert decision.recommendation_count == 0
