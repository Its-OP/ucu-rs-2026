"""Train and evaluate the two-stage hybrid recommender.

Usage examples
--------------
Global split (validation):
    python -m src.run_hybrid_two_stage --split-type global --split val --evaluator basic

Global split (test, offline diagnostics):
    python -m src.run_hybrid_two_stage --split-type global --split test --evaluator offline --ks 10,20

Per-user split:
    python -m src.run_hybrid_two_stage --split-type per_user --evaluator offline --ks 10,20
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from data.dataframes import (
    movies_enriched,
    test,
    train,
    user_based_temporal_train,
    user_based_temporal_val,
    users,
    val,
)
from src.eval.eval import evaluate as evaluate_basic
from src.eval.offline_ranking import evaluate as evaluate_offline
from src.models.hybrid_two_stage import TwoStageHybridRecommender

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate two-stage hybrid (BPR + content + reranker).",
    )

    parser.add_argument(
        "--split-type",
        type=str,
        default="global",
        choices=["global", "per_user"],
        help="Data split protocol: global temporal or per-user temporal.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Evaluation split. For per_user only 'val' is supported.",
    )
    parser.add_argument(
        "--evaluator",
        type=str,
        default="basic",
        choices=["basic", "offline"],
        help="Evaluation backend: basic (eval.py) or offline (offline_ranking.py).",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ks", type=str, default="10,20")
    parser.add_argument("--threshold", type=float, default=4.0)
    parser.add_argument("--mode", type=str, default="all", choices=["all", "warm_only"])

    parser.add_argument("--cf-n-factors", type=int, default=64)
    parser.add_argument("--cf-n-epochs", type=int, default=20)
    parser.add_argument("--cf-lr", type=float, default=0.01)
    parser.add_argument("--cf-regularization", type=float, default=0.01)
    parser.add_argument(
        "--cf-negative-sampling",
        type=str,
        default="uniform",
        choices=["uniform", "popularity"],
    )
    parser.add_argument(
        "--cf-negative-pool",
        type=str,
        default="unseen",
        choices=["unseen", "non_positive"],
    )
    parser.add_argument("--cf-popularity-alpha", type=float, default=0.75)

    parser.add_argument("--content-metric", type=str, default="pearson", choices=["cosine", "pearson"])
    parser.add_argument("--content-recency-decay", type=float, default=1.3)
    parser.add_argument("--content-n-neighbors", type=int, default=12)
    parser.add_argument("--content-min-liked", type=int, default=5)
    parser.add_argument("--content-min-ratings", type=int, default=100)

    parser.add_argument("--cf-candidates", type=int, default=200)
    parser.add_argument("--cb-candidates", type=int, default=200)
    parser.add_argument("--cb-search-size", type=int, default=400)
    parser.add_argument("--train-cf-candidates", type=int, default=120)
    parser.add_argument("--train-cb-candidates", type=int, default=120)
    parser.add_argument("--train-cb-search-size", type=int, default=240)
    parser.add_argument("--blend-alpha", type=float, default=0.7)
    parser.add_argument("--disable-ranker", action="store_true")
    parser.add_argument(
        "--ranker-cf-blend",
        type=float,
        default=1.0,
        help=(
            "Blend ranker score with normalised BPR CF score at prediction time. "
            "1.0 = pure ranker; 0.0 = pure CF; 0.7 = 70%% ranker + 30%% CF."
        ),
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rerank-holdout-frac",
        type=float,
        default=0.15,
        help="Per-user tail fraction used as reranker labels for *val* experiments.",
    )
    return parser.parse_args()


def _per_user_tail_holdout(
    ratings: pd.DataFrame,
    frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each user's timeline into base-train prefix and holdout tail."""
    if not (0.0 < frac < 0.5):
        raise ValueError("--rerank-holdout-frac must be in (0, 0.5)")

    base_parts: list[pd.DataFrame] = []
    holdout_parts: list[pd.DataFrame] = []
    for _, group in ratings.groupby("UserID"):
        group = group.sort_values("Timestamp")
        n = len(group)
        n_holdout = max(1, int(round(n * frac)))
        if n - n_holdout < 1:
            n_holdout = 1
        cut = n - n_holdout
        base_parts.append(group.iloc[:cut])
        holdout_parts.append(group.iloc[cut:])

    base_df = pd.concat(base_parts, ignore_index=True)
    holdout_df = pd.concat(holdout_parts, ignore_index=True)
    return base_df, holdout_df


def _resolve_split(
    split_type: str,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    if split_type == "global":
        if split == "test":
            # test protocol: fit base retrievers on train, train reranker on val labels
            return train.copy(), test.copy(), val.copy()
        # validation protocol: carve reranker labels from train to avoid leakage from val
        return train.copy(), val.copy(), None

    if split == "test":
        raise ValueError("per_user split currently supports only --split val")
    # per-user validation protocol: hold out a tail from train for reranker labels
    return user_based_temporal_train.copy(), user_based_temporal_val.copy(), None


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    train_ratings, eval_ratings, rerank_ratings = _resolve_split(args.split_type, args.split)
    if rerank_ratings is None and not args.disable_ranker:
        train_ratings, rerank_ratings = _per_user_tail_holdout(
            ratings=train_ratings,
            frac=args.rerank_holdout_frac,
        )
        logger.info(
            "Reranker holdout generated from training window: base_train=%d rerank_labels=%d",
            len(train_ratings),
            len(rerank_ratings),
        )

    model = TwoStageHybridRecommender(
        threshold=args.threshold,
        cf_n_factors=args.cf_n_factors,
        cf_n_epochs=args.cf_n_epochs,
        cf_lr=args.cf_lr,
        cf_regularization=args.cf_regularization,
        cf_negative_sampling=args.cf_negative_sampling,
        cf_negative_pool=args.cf_negative_pool,
        cf_popularity_alpha=args.cf_popularity_alpha,
        random_state=args.seed,
        content_metric=args.content_metric,
        content_recency_decay=args.content_recency_decay,
        content_n_neighbors=args.content_n_neighbors,
        content_min_liked=args.content_min_liked,
        content_min_ratings=args.content_min_ratings,
        cf_candidates=args.cf_candidates,
        cb_candidates=args.cb_candidates,
        cb_search_size=args.cb_search_size,
        train_cf_candidates=args.train_cf_candidates,
        train_cb_candidates=args.train_cb_candidates,
        train_cb_search_size=args.train_cb_search_size,
        use_ranker=not args.disable_ranker,
        blend_alpha=args.blend_alpha,
        ranker_cf_blend=args.ranker_cf_blend,
    )

    logger.info("Fitting TwoStageHybridRecommender...")
    model.fit(
        ratings=train_ratings,
        users=users,
        movies=movies_enriched,
        rerank_ratings=rerank_ratings,
    )

    if args.evaluator == "basic":
        metrics = evaluate_basic(
            model=model,
            train_ratings=train_ratings,
            test_ratings=eval_ratings,
            users=users,
            movies=movies_enriched,
            k=args.k,
            threshold=args.threshold,
        )
        logger.info("=== Basic Evaluation ===")
        logger.info("Split type: %s", args.split_type)
        logger.info("Split: %s", args.split)
        logger.info("NDCG@%d:      %.5f", args.k, metrics.ndcg)
        logger.info("Precision@%d: %.5f", args.k, metrics.precision)
        logger.info("Recall@%d:    %.5f", args.k, metrics.recall)
        return

    ks = [int(v.strip()) for v in args.ks.split(",") if v.strip()]
    report = evaluate_offline(
        model=model,
        train_ratings=train_ratings,
        test_ratings=eval_ratings,
        users=users,
        movies=movies_enriched,
        ks=ks,
        threshold=args.threshold,
        mode=args.mode,
    )
    logger.info("=== Offline Evaluation ===")
    logger.info("Split type: %s", args.split_type)
    logger.info("Split: %s", args.split)
    logger.info("Mode: %s", report.mode)
    logger.info(
        "Users total=%d, with_gt=%d, eligible=%d, evaluated=%d, skipped=%d",
        report.n_users_total,
        report.n_users_with_gt,
        report.n_users_eligible,
        report.n_users_evaluated,
        report.n_skipped,
    )
    logger.info(
        "Coverage rate=%.5f, avg list size=%.2f, skip rate=%.5f",
        report.coverage_rate,
        report.avg_list_size,
        report.skip_rate,
    )
    for k in sorted(report.by_k):
        m = report.by_k[k]
        logger.info(
            "k=%d: ndcg=%.5f precision=%.5f recall=%.5f mrr=%.5f map=%.5f",
            k,
            m.ndcg,
            m.precision,
            m.recall,
            m.mrr,
            m.map,
        )


if __name__ == "__main__":
    main()
