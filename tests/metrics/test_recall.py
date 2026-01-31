import pytest

from src.eval.metrics.recall import recall_at_k


def test_all_relevant_captured():
    # arrange
    ranked = [1, 2, 3]
    true = {1: 5.0, 2: 4.0, 3: 4.5}

    # act
    result = recall_at_k(ranked, true, k=3)

    # assert
    assert result == pytest.approx(1.0)


def test_no_relevant_items_in_truth():
    """When the user has no items >= threshold, recall is 0."""
    # arrange
    ranked = [1, 2, 3]
    true = {1: 2.0, 2: 3.0, 3: 1.0}

    # act
    result = recall_at_k(ranked, true, k=3)

    # assert
    assert result == pytest.approx(0.0)


def test_empty_truth():
    # arrange
    ranked = [1, 2, 3]
    true = {}

    # act
    result = recall_at_k(ranked, true, k=3)

    # assert
    assert result == pytest.approx(0.0)


def test_partial_recall():
    # arrange
    ranked = [1, 2, 3]
    true = {1: 5.0, 2: 2.0, 3: 4.0, 4: 5.0}

    # act
    result = recall_at_k(ranked, true, k=3)

    # assert — relevant: {1, 3, 4}, captured: {1, 3}
    assert result == pytest.approx(2 / 3)


def test_relevant_items_outside_top_k():
    # arrange
    ranked = [10, 20, 1, 2]
    true = {1: 5.0, 2: 5.0}

    # act
    result = recall_at_k(ranked, true, k=2)

    # assert — k=2: only items 10, 20 checked, neither relevant
    assert result == pytest.approx(0.0)


def test_k_truncates():
    # arrange
    ranked = [1, 2, 3, 4]
    true = {1: 5.0, 3: 4.0}

    # act
    result_k1 = recall_at_k(ranked, true, k=1)
    result_k3 = recall_at_k(ranked, true, k=3)

    # assert — k=1: {1} captured out of {1,3} → 1/2; k=3: {1,3} → 2/2
    assert result_k1 == pytest.approx(0.5)
    assert result_k3 == pytest.approx(1.0)


def test_unknown_items_are_non_relevant():
    # arrange
    ranked = [99, 88, 1]
    true = {1: 5.0, 2: 4.0}

    # act
    result = recall_at_k(ranked, true, k=3)

    # assert — relevant: {1, 2}, captured: {1}
    assert result == pytest.approx(0.5)


def test_custom_threshold():
    # arrange
    ranked = [1, 2, 3]
    true = {1: 3.0, 2: 3.5, 3: 2.0}

    # act
    result_default = recall_at_k(ranked, true, k=3)
    result_custom = recall_at_k(ranked, true, k=3, threshold=3.0)

    # assert — default: 0 relevant → 0.0; threshold=3: {1,2} captured → 1.0
    assert result_default == pytest.approx(0.0)
    assert result_custom == pytest.approx(1.0)


def test_k_larger_than_list():
    # arrange
    ranked = [1]
    true = {1: 5.0, 2: 5.0}

    # act
    result = recall_at_k(ranked, true, k=10)

    # assert — k=10 but only 1 item ranked, captured {1} out of {1,2}
    assert result == pytest.approx(0.5)


def test_boundary_rating_is_relevant():
    # arrange
    ranked = [1]
    true = {1: 4.0}

    # act
    result = recall_at_k(ranked, true, k=1)

    # assert
    assert result == pytest.approx(1.0)


def test_accepts_list_input():
    # arrange
    ranked = [1, 2]
    true = {1: 5.0, 2: 4.0}

    # act
    result = recall_at_k(ranked, true, k=2)

    # assert
    assert result == pytest.approx(1.0)
