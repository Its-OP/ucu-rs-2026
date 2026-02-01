import pandas as pd
import pytest

from src.eval.eval import Metrics, evaluate
from src.models.base import Rating, RecommenderModel


class PerfectModel(RecommenderModel):
    """Return items sorted by their true test rating."""

    def __init__(self, test_ratings: pd.DataFrame):
        self._test = test_ratings

    def predict(self, users, ratings, movies, k=10):
        result = {}
        for user_id, group in self._test.groupby("UserID"):
            top = group.nlargest(k, "Rating")
            result[user_id] = [
                Rating(movie_id=row.MovieID, score=row.Rating)
                for row in top.itertuples()
            ]
        return result


class PopularityModel(RecommenderModel):
    """Always recommend globally most-rated movies."""

    def __init__(self, all_ratings: pd.DataFrame):
        self._popular = (
            all_ratings.groupby("MovieID")["Rating"]
            .count()
            .nlargest(100)
            .index
            .tolist()
        )

    def predict(self, users, ratings, movies, k=10):
        result = {}
        popular_recs = [
            Rating(movie_id=mid, score=float(100 - i))
            for i, mid in enumerate(self._popular[:k])
        ]
        for user_id in users["UserID"].unique():
            result[user_id] = popular_recs
        return result


@pytest.fixture
def sample_users():
    return pd.DataFrame({
        "UserID": [1, 2, 3],
        "Gender": ["M", "F", "M"],
        "Age": [25, 35, 18],
        "Occupation": [0, 1, 2],
        "Zip-code": ["00000", "11111", "22222"],
    })


@pytest.fixture
def sample_movies():
    return pd.DataFrame({
        "MovieID": [10, 20, 30, 40, 50],
        "Title": ["A", "B", "C", "D", "E"],
        "Genres": ["Action", "Comedy", "Drama", "Action", "Comedy"],
    })


@pytest.fixture
def sample_train_ratings():
    return pd.DataFrame({
        "UserID":  [1, 1, 2, 2, 3, 3],
        "MovieID": [10, 20, 30, 40, 10, 50],
        "Rating":  [5, 3, 4, 2, 4, 5],
        "Timestamp": pd.to_datetime([
            "2000-06-01", "2000-06-02", "2000-06-03",
            "2000-06-04", "2000-06-05", "2000-06-06",
        ]),
    })


@pytest.fixture
def sample_test_ratings():
    return pd.DataFrame({
        "UserID":  [1, 1, 2, 2, 3, 3],
        "MovieID": [30, 40, 10, 50, 20, 40],
        "Rating":  [5, 4, 5, 1, 3, 5],
        "Timestamp": pd.to_datetime([
            "2000-12-01", "2000-12-02", "2000-12-03",
            "2000-12-04", "2000-12-05", "2000-12-06",
        ]),
    })


def test_perfect_model_achieves_ideal_ndcg(
    sample_users, sample_movies, sample_train_ratings, sample_test_ratings,
):
    # arrange
    model = PerfectModel(sample_test_ratings)

    # act
    result = evaluate(
        model=model,
        train_ratings=sample_train_ratings,
        test_ratings=sample_test_ratings,
        users=sample_users,
        movies=sample_movies,
        k=2,
    )

    # assert
    assert result.ndcg == pytest.approx(1.0)


def test_perfect_model_precision_and_recall(
    sample_users, sample_movies, sample_train_ratings, sample_test_ratings,
):
    # arrange
    model = PerfectModel(sample_test_ratings)

    # act
    result = evaluate(
        model=model,
        train_ratings=sample_train_ratings,
        test_ratings=sample_test_ratings,
        users=sample_users,
        movies=sample_movies,
        k=2,
    )

    # assert
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)


def test_popularity_model_metrics(
    sample_users, sample_movies, sample_train_ratings, sample_test_ratings,
):
    """Popularity baseline recommends [10, 20, ...] to everyone.
    User 1: 2 relevant, effective_k=2, recs=[10,20], 0 hits -> prec=0, recall=0
    User 2: 1 relevant, effective_k=1, top-1=[10], hit -> prec=1, recall=1
    User 3: 1 relevant, effective_k=1, top-1=[10], miss -> prec=0, recall=0"""
    # arrange
    model = PopularityModel(sample_train_ratings)

    # act
    result = evaluate(
        model=model,
        train_ratings=sample_train_ratings,
        test_ratings=sample_test_ratings,
        users=sample_users,
        movies=sample_movies,
        k=2,
    )

    # assert
    assert result.precision == pytest.approx(1 / 3)
    assert result.recall == pytest.approx(1 / 3)


def test_strict_threshold_lowers_precision(
    sample_movies, sample_train_ratings,
):
    # arrange
    users = pd.DataFrame({
        "UserID": [1],
        "Gender": ["M"],
        "Age": [25],
        "Occupation": [0],
        "Zip-code": ["00000"],
    })
    test_ratings = pd.DataFrame({
        "UserID":  [1, 1],
        "MovieID": [30, 40],
        "Rating":  [5.0, 4.0],
        "Timestamp": pd.to_datetime(["2000-12-01", "2000-12-02"]),
    })

    class FixedModel(RecommenderModel):
        def predict(self, users, ratings, movies, k=10):
            return {1: [Rating(movie_id=40, score=2.0), Rating(movie_id=30, score=1.0)]}

    model = FixedModel()

    # act
    result_lenient = evaluate(
        model=model,
        train_ratings=sample_train_ratings,
        test_ratings=test_ratings,
        users=users,
        movies=sample_movies,
        k=2,
        threshold=4.0,
    )
    result_strict = evaluate(
        model=model,
        train_ratings=sample_train_ratings,
        test_ratings=test_ratings,
        users=users,
        movies=sample_movies,
        k=2,
        threshold=5.0,
    )

    # assert
    assert result_lenient.precision == pytest.approx(1.0)
    assert result_strict.precision == pytest.approx(0.0)
