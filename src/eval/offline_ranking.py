import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Tuple, TypeVar

import numpy as np
import pandas as pd

from src.eval.metrics.ndcg import ndcg_at_k
from src.eval.metrics.precision import precision_at_k
from src.eval.metrics.recall import recall_at_k
from src.models.base import RecommenderModel

logger = logging.getLogger(__name__)

TRec = TypeVar("TRec")


@dataclass(frozen=True, kw_only=True)
class MetricsAtK:
    k: int
    ndcg: float
    precision: float
    recall: float
    mrr: float


@dataclass(frozen=True, kw_only=True)
class EvalReport:
    by_k: Dict[int, MetricsAtK]

    n_users_total: int
    n_users_with_gt: int
    n_users_eligible: int
    n_users_evaluated: int
    n_predicted: int
    n_skipped: int
    skip_rate: float

    coverage_rate: float
    avg_list_size: float

    mode: Literal["all", "warm_only"]
    n_warm_users: int
    n_cold_users: int
    cold_user_rate: float


def mrr_at_k(
    ranked_item_ids: np.ndarray,
    true_ratings: Dict[int, float],
    k: int,
    threshold: float,
) -> Tuple[float, bool]:
    """Compute reciprocal rank at K for a single user.

    Parameters
    ----------
    ranked_item_ids : np.ndarray
        Ranked item identifiers returned by the model (best first).
    true_ratings : dict[int, float]
        Ground-truth ratings for the user (item_id -> rating).
    k : int
        Cut-off position.
    threshold : float
        Binary relevance threshold (rating >= threshold is relevant).

    Returns
    -------
    tuple[float, bool]
        Reciprocal rank value and an eligibility flag. If the user has no relevant items
        in ground truth under ``threshold``, returns (0.0, False).
    """
    relevant = {iid for iid, r in true_ratings.items() if r >= threshold}
    if not relevant:
        return 0.0, False

    topk = ranked_item_ids[:k]
    for rank, iid in enumerate(topk, start=1):
        if int(iid) in relevant:
            return 1.0 / rank, True
    return 0.0, True


def _compute_warm_cold_users(
    users_df: pd.DataFrame,
    train_ratings: pd.DataFrame,
    threshold: float,
    user_col: str,
    rating_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split users into warm and cold segments based on training interactions.

    A user is considered warm if they have at least one relevant interaction in the
    training data (rating >= threshold). Otherwise, they are considered cold.

    Parameters
    ----------
    users_df : pd.DataFrame
        User table containing user identifiers.
    train_ratings : pd.DataFrame
        Training interactions.
    threshold : float
        Relevance threshold used to define positive interactions.
    user_col : str
        Column name for user identifiers.
    rating_col : str
        Column name for rating values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays of warm user IDs and cold user IDs.
    """
    user_ids = users_df[user_col].to_numpy()

    train_pos = train_ratings.loc[train_ratings[rating_col] >= threshold, [user_col]]
    warm_set = set(train_pos[user_col].unique().tolist())

    warm_mask = np.array([u in warm_set for u in user_ids], dtype=bool)
    return user_ids[warm_mask], user_ids[~warm_mask]


def evaluate(
    model: RecommenderModel,
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    ks: Iterable[int] = (10, 20),
    threshold: float = 4.0,
    mode: Literal["all", "warm_only"] = "all",
    user_col: str = "UserID",
    item_col: str = "MovieID",
    rating_col: str = "Rating",
) -> EvalReport:
    """Evaluate a recommender model on a held-out test set using ranking metrics.

    This evaluator extends the basic offline protocol with:
    - multiple cut-offs (K values),
    - additional ranking metric (MRR@K),
    - warm-users evaluation mode,
    - user accounting (how many users were eligible/evaluated),
    - coverage statistics (whether the model returns enough recommendations).

    Parameters
    ----------
    model : RecommenderModel
        A fitted model exposing a ``predict`` method.
    train_ratings : pd.DataFrame
        Training interactions passed to the model for prediction and warm/cold segmentation.
    test_ratings : pd.DataFrame
        Held-out interactions used as ground truth.
    users : pd.DataFrame
        User side-information. The set of users to evaluate is derived from this table
        and optionally filtered by ``mode``.
    movies : pd.DataFrame
        Item side-information.
    ks : Iterable[int], default=(10, 20)
        Cut-off positions for all metrics. The model is asked for ``max(ks)`` recommendations.
    threshold : float, default=4.0
        Binary relevance threshold for Precision@K, Recall@K, and MRR@K
        (items with rating >= threshold are treated as relevant).
    mode : {"all", "warm_only"}, default="all"
        Evaluation user subset:
        - ``"all"`` evaluates all users in ``users``.
        - ``"warm_only"`` evaluates only users with at least one relevant interaction
          in ``train_ratings`` under the given ``threshold``.
    user_col : str, default="UserID"
        Column name for user identifiers in ``train_ratings``, ``test_ratings``, and ``users``.
    item_col : str, default="MovieID"
        Column name for item identifiers in ``train_ratings`` and ``test_ratings``.
    rating_col : str, default="Rating"
        Column name for ratings in ``train_ratings`` and ``test_ratings``.

    Returns
    -------
    EvalReport
        Evaluation report containing:
        - aggregated metrics per K (NDCG@K, Precision@K, Recall@K, MRR@K),
        - user counts (total/with ground truth/eligible/evaluated),
        - skip rate (users skipped due to having no relevant items in the test set),
        - coverage statistics,
        - warm/cold user counts and ``cold_user_rate``.

    Notes
    -----
    Users with no relevant items in the test set under ``threshold`` are excluded from
    metric averaging (they are counted in ``n_skipped`` and reflected in ``skip_rate``).
    """
    ks = sorted(set(int(k) for k in ks))
    if not ks:
        raise ValueError("ks must be a non-empty iterable of ints")
    k_max = max(ks)
    k_min = min(ks)

    warm_users, cold_users = _compute_warm_cold_users(
        users_df=users,
        train_ratings=train_ratings,
        threshold=threshold,
        user_col=user_col,
        rating_col=rating_col,
    )

    n_users_total = int(users[user_col].nunique())
    n_warm_users = int(len(warm_users))
    n_cold_users = int(len(cold_users))
    cold_user_rate = float(n_cold_users / n_users_total) if n_users_total else 0.0

    if mode == "warm_only":
        users_eval = users[users[user_col].isin(warm_users)].copy()
    elif mode == "all":
        users_eval = users.copy()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    predictions: Dict[int, list[TRec]] = model.predict(
        users_eval, train_ratings, movies, k=k_max
    )
    n_predicted = int(len(predictions))

    ground_truth: Dict[int, Dict[int, float]] = {
        int(uid): dict(zip(g[item_col].astype(int), g[rating_col].astype(float)))
        for uid, g in test_ratings.groupby(user_col)
    }

    users_eval_ids = users_eval[user_col].astype(int).unique()
    n_users_with_gt = int(sum(int(uid) in ground_truth for uid in users_eval_ids))

    def _is_eligible(uid: int) -> bool:
        gt = ground_truth.get(int(uid), {})
        return any(r >= threshold for r in gt.values())

    n_users_eligible = int(sum(_is_eligible(uid) for uid in users_eval_ids))

    ndcg_scores: Dict[int, list] = {k: [] for k in ks}
    precision_scores: Dict[int, list] = {k: [] for k in ks}
    recall_scores: Dict[int, list] = {k: [] for k in ks}
    mrr_scores: Dict[int, list] = {k: [] for k in ks}

    list_sizes = []
    n_users_with_min_k = 0

    n_skipped = 0
    n_evaluated = 0

    for uid, recs in predictions.items():
        uid = int(uid)
        recs_list = list(recs) if recs is not None else []

        list_sizes.append(len(recs_list))
        if len(recs_list) >= k_min:
            n_users_with_min_k += 1

        true_ratings = ground_truth.get(uid, {})
        if not any(r >= threshold for r in true_ratings.values()):
            n_skipped += 1
            continue

        ranked_item_ids = np.array(
            [int(getattr(r, "movie_id")) for r in recs_list], dtype=int
        )

        for k in ks:
            nd = ndcg_at_k(ranked_item_ids, true_ratings, k=k)
            pr = precision_at_k(ranked_item_ids, true_ratings, k=k, threshold=threshold)
            rc = recall_at_k(ranked_item_ids, true_ratings, k=k, threshold=threshold)
            mr, eligible = mrr_at_k(
                ranked_item_ids, true_ratings, k=k, threshold=threshold
            )

            if pr is None or rc is None or not eligible:
                n_skipped += 1
                break

            ndcg_scores[k].append(float(nd))
            precision_scores[k].append(float(pr))
            recall_scores[k].append(float(rc))
            mrr_scores[k].append(float(mr))
        else:
            n_evaluated += 1

    skip_rate = float(n_skipped / n_predicted) if n_predicted else 0.0
    if n_skipped:
        logger.warning(
            "Skipped %d/%d users with no relevant items in test set (threshold=%.2f)",
            n_skipped,
            n_predicted,
            threshold,
        )

    avg_list_size = float(np.mean(list_sizes)) if list_sizes else 0.0
    coverage_rate = float(n_users_with_min_k / n_predicted) if n_predicted else 0.0

    by_k: Dict[int, MetricsAtK] = {}
    for k in ks:
        by_k[k] = MetricsAtK(
            k=k,
            ndcg=float(np.mean(ndcg_scores[k])) if ndcg_scores[k] else 0.0,
            precision=(
                float(np.mean(precision_scores[k])) if precision_scores[k] else 0.0
            ),
            recall=float(np.mean(recall_scores[k])) if recall_scores[k] else 0.0,
            mrr=float(np.mean(mrr_scores[k])) if mrr_scores[k] else 0.0,
        )

    return EvalReport(
        by_k=by_k,
        n_users_total=n_users_total,
        n_users_with_gt=n_users_with_gt,
        n_users_eligible=n_users_eligible,
        n_users_evaluated=int(n_evaluated),
        n_predicted=n_predicted,
        n_skipped=int(n_skipped),
        skip_rate=skip_rate,
        coverage_rate=coverage_rate,
        avg_list_size=avg_list_size,
        mode=mode,
        n_warm_users=n_warm_users,
        n_cold_users=n_cold_users,
        cold_user_rate=cold_user_rate,
    )
