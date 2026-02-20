"""Train and evaluate Bayesian Personalized Ranking (BPR-OPT).

Usage examples
--------------
Tune on validation split:
    python -m src.run_bpr --split val --evaluator basic

Final test evaluation:
    python -m src.run_bpr --split test --evaluator basic

Offline ranking report:
    python -m src.run_bpr --evaluator offline --ks 10,20 --mode all
"""

from __future__ import annotations

import argparse
import logging

from data.dataframes import movies, test, train, users, val
from src.eval.eval import evaluate as evaluate_basic
from src.eval.offline_ranking import evaluate as evaluate_offline
from src.models.bpr import BPRRecommender

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate BPR recommender.")

    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--evaluator", type=str, default="basic", choices=["basic", "offline"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ks", type=str, default="10,20")
    parser.add_argument("--threshold", type=float, default=4.0)
    parser.add_argument("--mode", type=str, default="all", choices=["all", "warm_only"])

    parser.add_argument("--n-factors", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--n-samples-per-epoch", type=int, default=0)
    parser.add_argument("--positive-threshold", type=float, default=4.0)
    parser.add_argument(
        "--negative-sampling",
        type=str,
        default="uniform",
        choices=["uniform", "popularity"],
    )
    parser.add_argument(
        "--negative-pool",
        type=str,
        default="unseen",
        choices=["unseen", "non_positive"],
        help=(
            "Negative candidate pool: unseen (exclude all train-seen items) "
            "or non_positive (exclude only positives)."
        ),
    )
    parser.add_argument("--popularity-alpha", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    model = BPRRecommender(
        n_factors=args.n_factors,
        n_epochs=args.n_epochs,
        lr=args.lr,
        regularization=args.regularization, 
        n_samples_per_epoch=(
            None if args.n_samples_per_epoch <= 0 else args.n_samples_per_epoch
        ),
        threshold=args.positive_threshold,
        negative_sampling=args.negative_sampling,
        negative_pool=args.negative_pool,
        popularity_alpha=args.popularity_alpha,
        random_state=args.seed,
    )

    eval_ratings = val if args.split == "val" else test

    logger.info("Fitting BPR on train split...")
    model.fit(train, users=users, movies=movies)

    if args.evaluator == "basic":
        metrics = evaluate_basic(
            model=model,
            train_ratings=train,
            test_ratings=eval_ratings,
            users=users,
            movies=movies,
            k=args.k,
            threshold=args.threshold,
        )
        logger.info("=== Basic Evaluation ===")
        logger.info("Split: %s", args.split)
        logger.info("NDCG@%d:      %.5f", args.k, metrics.ndcg)
        logger.info("Precision@%d: %.5f", args.k, metrics.precision)
        logger.info("Recall@%d:    %.5f", args.k, metrics.recall)
        return

    ks = [int(v.strip()) for v in args.ks.split(",") if v.strip()]
    report = evaluate_offline(
        model=model,
        train_ratings=train,
        test_ratings=eval_ratings,
        users=users,
        movies=movies,
        ks=ks,
        threshold=args.threshold,
        mode=args.mode,
    )
    logger.info("=== Offline Evaluation ===")
    logger.info("Split: %s", args.split)
    logger.info(
        "Users total=%d, with_gt=%d, eligible=%d, evaluated=%d, skipped=%d",
        report.n_users_total,
        report.n_users_with_gt,
        report.n_users_eligible,
        report.n_users_evaluated,
        report.n_skipped,
    )
    for k in sorted(report.by_k):
        m = report.by_k[k]
        logger.info(
            "k=%d: ndcg=%.5f precision=%.5f recall=%.5f mrr=%.5f",
            k,
            m.ndcg,
            m.precision,
            m.recall,
            m.mrr,
        )


if __name__ == "__main__":
    main()
