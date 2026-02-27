"""Train and evaluate a Wide & Deep recommender.

Usage examples
--------------
Tune on validation split:
    python -m src.run_wide_deep --split val --evaluator offline

Final test evaluation:
    python -m src.run_wide_deep --split test --evaluator offline
"""

from __future__ import annotations

import argparse
import logging

from data.dataframes import movies, test, train, users, val
from src.eval.eval import evaluate as evaluate_basic
from src.eval.offline_ranking import evaluate as evaluate_offline
from src.models.wide_deep import WideAndDeepRecommender

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate Wide&Deep recommender.")

    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--evaluator", type=str, default="offline", choices=["basic", "offline"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ks", type=str, default="10,20")
    parser.add_argument("--threshold", type=float, default=4.0)
    parser.add_argument("--mode", type=str, default="all", choices=["all", "warm_only"])

    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--n-negatives", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dims", type=str, default="128,64")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--genre-embedding-dim", type=int, default=16)
    parser.add_argument("--max-positive-samples-per-epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    hidden_dims = tuple(int(v.strip()) for v in args.hidden_dims.split(",") if v.strip())
    if len(hidden_dims) != 2:
        raise ValueError("--hidden-dims must contain exactly 2 comma-separated integers")

    model = WideAndDeepRecommender(
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        threshold=args.threshold,
        n_negatives=args.n_negatives,
        embedding_dim=args.embedding_dim,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        genre_embedding_dim=args.genre_embedding_dim,
        max_positive_samples_per_epoch=args.max_positive_samples_per_epoch,
        random_state=args.seed,
        device=args.device,
    )

    eval_ratings = val if args.split == "val" else test

    logger.info("Fitting Wide&Deep on global train split...")
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
        logger.info("NDCG@%d: %.5f", args.k, metrics.ndcg)
        logger.info("Precision@%d: %.5f", args.k, metrics.precision)
        logger.info("Recall@%d: %.5f", args.k, metrics.recall)
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
