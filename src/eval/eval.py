import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.eval.metrics.ndcg import ndcg_at_k
from src.eval.metrics.precision import precision_at_k
from src.eval.metrics.recall import recall_at_k
from src.models.base import RecommenderModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class Metrics:
    ndcg: float
    precision: float
    recall: float


def evaluate(
    model: RecommenderModel,
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    k: int = 10,
    threshold: float = 4.0,
) -> Metrics:
    """Evaluate a recommender model on a held-out test set.

    Parameters
    ----------
    model : RecommenderModel
        A fitted model exposing a ``predict`` method.
    train_ratings : pd.DataFrame
        Training interactions passed to the model for prediction.
    test_ratings : pd.DataFrame
        Held-out interactions used as ground truth.
    users : pd.DataFrame
        User side-information.
    movies : pd.DataFrame
        Movie side-information.
    k : int
        Cut-off position for all metrics.
    threshold : float
        Binary relevance threshold for Precision@K and Recall@K.

    Returns
    -------
    Metrics
        Aggregated NDCG@K, Precision@K, and Recall@K averaged across users.
    """
    predictions = model.predict(users, train_ratings, movies, k=k)

    ground_truth = {
        user_id: dict(zip(group["MovieID"], group["Rating"]))
        for user_id, group in test_ratings.groupby("UserID")
    }

    ndcg_scores = []
    precision_scores = []
    recall_scores = []
    n_skipped = 0

    for user_id, rated_items in predictions.items():
        true_ratings = ground_truth.get(user_id, {})

        ranked_item_ids = np.array([r.movie_id for r in rated_items])

        ndcg = ndcg_at_k(ranked_item_ids, true_ratings, k=k)
        precision = precision_at_k(
            ranked_item_ids, true_ratings, k=k, threshold=threshold
        )
        recall = recall_at_k(ranked_item_ids, true_ratings, k=k, threshold=threshold)

        if precision is None or recall is None:
            n_skipped += 1
            continue

        ndcg_scores.append(ndcg)
        precision_scores.append(precision)
        recall_scores.append(recall)

    if n_skipped > 0:
        logger.warning(
            "Skipped %d/%d users with no relevant items in test set",
            n_skipped,
            len(predictions),
        )

    return Metrics(
        ndcg=float(np.mean(ndcg_scores)),
        precision=float(np.mean(precision_scores)),
        recall=float(np.mean(recall_scores)),
    )
