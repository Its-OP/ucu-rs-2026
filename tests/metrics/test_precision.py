import numpy as np
import pytest

from src.eval.metrics.precision import precision_at_k


def test_all_relevant():
    # arrange
    ranked_item_ids = np.array([1, 2, 3])
    true_ratings = {1: 5.0, 2: 4.0, 3: 4.5}

    # act
    result = precision_at_k(ranked_item_ids, true_ratings, k=3)

    # assert
    assert result == pytest.approx(1.0)


def test_none_relevant():
    # arrange
    ranked_item_ids = np.array([1, 2, 3])
    true_ratings = {1: 2.0, 2: 3.0, 3: 1.0}

    # act
    result = precision_at_k(ranked_item_ids, true_ratings, k=3)

    # assert — no relevant items → None
    assert result is None


def test_partial_hits():
    # arrange
    ranked_item_ids = np.array([1, 2, 3, 4])
    true_ratings = {1: 5.0, 2: 2.0, 3: 4.0, 4: 1.0}

    # act
    result = precision_at_k(ranked_item_ids, true_ratings, k=4)

    # assert — 2 out of 4 are >= 4
    assert result == pytest.approx(0.5)


def test_unknown_items_are_non_relevant():
    # arrange
    ranked_item_ids = np.array([99, 88, 1])
    true_ratings = {1: 5.0}

    # act
    result = precision_at_k(ranked_item_ids, true_ratings, k=3)

    # assert — 1 relevant item, effective_k=1, top-1 is 99 (miss) → 0.0
    assert result == pytest.approx(0.0)


def test_k_truncates():
    # arrange
    ranked_item_ids = np.array([1, 2, 3, 4])
    true_ratings = {1: 5.0, 2: 2.0, 3: 5.0, 4: 5.0}

    # act
    result = precision_at_k(ranked_item_ids, true_ratings, k=2)

    # assert — k=2: only items 1 (hit) and 2 (miss)
    assert result == pytest.approx(0.5)


def test_custom_threshold():
    # arrange
    ranked_item_ids = np.array([1, 2, 3])
    true_ratings = {1: 3.0, 2: 3.5, 3: 3.2}

    # act
    result_default = precision_at_k(ranked_item_ids, true_ratings, k=3)
    result_custom = precision_at_k(ranked_item_ids, true_ratings, k=3, threshold=3.0)

    # assert
    assert result_default is None
    assert result_custom == pytest.approx(1.0)


def test_k_larger_than_list():
    # arrange
    ranked_item_ids = np.array([1, 2])
    true_ratings = {1: 5.0, 2: 5.0}

    # act
    result = precision_at_k(ranked_item_ids, true_ratings, k=5)

    # assert — 2 relevant, effective_k=min(5,2)=2, both hit → 1.0
    assert result == pytest.approx(1.0)


def test_boundary_rating_is_relevant():
    # arrange
    ranked_item_ids = np.array([1])
    true_ratings = {1: 4.0}

    # act
    result = precision_at_k(ranked_item_ids, true_ratings, k=1)

    # assert 4 === 4
    assert result == pytest.approx(1.0)


def test_just_below_threshold_is_not_relevant():
    # arrange
    ranked_item_ids = np.array([1])
    true_ratings = {1: 3.99}

    # act
    result = precision_at_k(ranked_item_ids, true_ratings, k=1)

    # assert 3.99 < 4
    assert result is None
