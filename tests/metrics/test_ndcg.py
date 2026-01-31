import math

import numpy as np
import pytest

from src.eval.metrics.ndcg import ndcg_at_k


def test_perfect_ranking():
    """Items ranked exactly by descending rating should yield NDCG = 1."""
    # arrange
    ranked = [10, 20, 30]
    true = {10: 5.0, 20: 4.0, 30: 3.0}

    # act
    result = ndcg_at_k(ranked, true, k=3)

    # assert
    assert result == pytest.approx(1.0)


def test_reversed_ranking():
    """Worst-case ordering: lowest-rated item first."""
    # arrange
    ranked = [30, 20, 10]
    true = {10: 5.0, 20: 4.0, 30: 3.0}

    # act
    result = ndcg_at_k(ranked, true, k=3)

    # assert
    assert 0.0 < result < 1.0


def test_single_relevant_item_at_top():
    # arrange
    ranked = [1, 2, 3]
    true = {1: 5.0}

    # act
    result = ndcg_at_k(ranked, true, k=3)

    # assert
    assert result == pytest.approx(1.0)


def test_single_relevant_item_not_at_top():
    # arrange
    ranked = [2, 1, 3]
    true = {1: 5.0}
    expected = (5 / math.log2(3)) / (5 / math.log2(2))

    # act
    result = ndcg_at_k(ranked, true, k=3)

    # assert — DCG = 5/log2(3), IDCG = 5/log2(2)
    assert result == pytest.approx(expected)


def test_no_relevant_items():
    """When the user has no rated items, NDCG should be 0."""
    # arrange
    ranked = [1, 2, 3]
    true = {}

    # act
    result = ndcg_at_k(ranked, true, k=3)

    # assert
    assert result == 0.0


def test_k_larger_than_ranked_list():
    """k exceeds the number of ranked items — should not error."""
    # arrange
    ranked = [1, 2]
    true = {1: 5.0, 2: 4.0}

    # act
    result = ndcg_at_k(ranked, true, k=10)

    # assert
    assert result == pytest.approx(1.0)


def test_k_truncates_ranked_list():
    """Only the first k items should matter."""
    # arrange
    ranked = [10, 20, 30, 40]
    true = {10: 5.0, 20: 4.0, 30: 1.0, 40: 5.0}
    # At k=2, ranked gets [10(5.0), 20(4.0)].
    # IDCG picks the best 2 from truth: 5.0 and 5.0 (items 10, 40).
    dcg = 5 / math.log2(2) + 4 / math.log2(3)
    idcg = 5 / math.log2(2) + 5 / math.log2(3)

    # act
    result = ndcg_at_k(ranked, true, k=2)

    # assert
    assert result == pytest.approx(dcg / idcg)


def test_unknown_items_have_zero_relevance():
    """Items not in true_ratings contribute 0 gain."""
    # arrange
    ranked = [99, 1]
    true = {1: 5.0}
    expected = (5 / math.log2(3)) / (5 / math.log2(2))

    # act
    result = ndcg_at_k(ranked, true, k=2)

    # assert — DCG = 0/log2(2) + 5/log2(3), IDCG = 5/log2(2)
    assert result == pytest.approx(expected)


def test_graded_relevance_distinguishes_ratings():
    """A 5-star item at rank 1 should score higher than a 3-star item at rank 1."""
    # arrange
    true = {1: 5.0, 2: 3.0}

    # act
    ndcg_good = ndcg_at_k([1, 2], true, k=2)
    ndcg_bad = ndcg_at_k([2, 1], true, k=2)

    # assert
    assert ndcg_good > ndcg_bad


def test_accepts_list_input():
    """Should accept plain Python lists, not just numpy arrays."""
    # arrange
    ranked = [1, 2, 3]
    true = {1: 5.0, 2: 4.0, 3: 3.0}

    # act
    result = ndcg_at_k(ranked, true, k=3)

    # assert
    assert result == pytest.approx(1.0)


def test_hand_computed_value():
    """Verify against a manually computed DCG/IDCG."""
    # arrange
    ranked = [3, 1, 2]
    true = {1: 5.0, 2: 4.0, 3: 2.0}
    dcg = 2 / math.log2(2) + 5 / math.log2(3) + 4 / math.log2(4)
    idcg = 5 / math.log2(2) + 4 / math.log2(3) + 2 / math.log2(4)

    # act
    result = ndcg_at_k(ranked, true, k=3)

    # assert
    assert result == pytest.approx(dcg / idcg)
