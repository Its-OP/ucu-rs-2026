import pytest

from src.eval.metrics.precision import precision_at_k


def test_all_relevant():
    # arrange
    ranked = [1, 2, 3]
    true = {1: 5.0, 2: 4.0, 3: 4.5}

    # act
    result = precision_at_k(ranked, true, k=3)

    # assert
    assert result == pytest.approx(1.0)


def test_none_relevant():
    # arrange
    ranked = [1, 2, 3]
    true = {1: 2.0, 2: 3.0, 3: 1.0}

    # act
    result = precision_at_k(ranked, true, k=3)

    # assert
    assert result == pytest.approx(0.0)


def test_partial_hits():
    # arrange
    ranked = [1, 2, 3, 4]
    true = {1: 5.0, 2: 2.0, 3: 4.0, 4: 1.0}

    # act
    result = precision_at_k(ranked, true, k=4)

    # assert — 2 out of 4 are >= 4
    assert result == pytest.approx(0.5)


def test_unknown_items_are_non_relevant():
    # arrange
    ranked = [99, 88, 1]
    true = {1: 5.0}

    # act
    result = precision_at_k(ranked, true, k=3)

    # assert — only item 1 is relevant, 99 and 88 are unknown
    assert result == pytest.approx(1 / 3)


def test_k_truncates():
    # arrange
    ranked = [1, 2, 3, 4]
    true = {1: 5.0, 2: 2.0, 3: 5.0, 4: 5.0}

    # act
    result = precision_at_k(ranked, true, k=2)

    # assert — k=2: only items 1 (hit) and 2 (miss)
    assert result == pytest.approx(0.5)


def test_custom_threshold():
    # arrange
    ranked = [1, 2, 3]
    true = {1: 3.0, 2: 3.5, 3: 2.0}

    # act
    result_default = precision_at_k(ranked, true, k=3)
    result_custom = precision_at_k(ranked, true, k=3, threshold=3.0)

    # assert — default threshold=4 gives 0 hits; threshold=3 gives 2/3
    assert result_default == pytest.approx(0.0)
    assert result_custom == pytest.approx(2 / 3)


def test_k_larger_than_list():
    # arrange
    ranked = [1, 2]
    true = {1: 5.0, 2: 5.0}

    # act
    result = precision_at_k(ranked, true, k=5)

    # assert — k=5 but only 2 items → 2/5
    assert result == pytest.approx(2 / 5)


def test_accepts_list_input():
    # arrange
    ranked = [1, 2]
    true = {1: 5.0, 2: 5.0}

    # act
    result = precision_at_k(ranked, true, k=2)

    # assert
    assert result == pytest.approx(1.0)


def test_boundary_rating_is_relevant():
    """A rating exactly at threshold should count as relevant."""
    # arrange
    ranked = [1]
    true = {1: 4.0}

    # act
    result = precision_at_k(ranked, true, k=1)

    # assert
    assert result == pytest.approx(1.0)


def test_just_below_threshold_is_not_relevant():
    # arrange
    ranked = [1]
    true = {1: 3.99}

    # act
    result = precision_at_k(ranked, true, k=1)

    # assert
    assert result == pytest.approx(0.0)
