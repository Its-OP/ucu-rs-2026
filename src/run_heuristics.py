"""Train and evaluate heuristic recommenders (popularity + graph-based).

This CLI is intended for fast baseline experiments and protocol checks.

Usage examples
--------------
Popularity heuristics:
    python -m src.run_heuristics --model count
    python -m src.run_heuristics --model bayesian --bayesian-m 50
    python -m src.run_heuristics --model recency --half-life-days 14

Graph heuristics:
    python -m src.run_heuristics --model item_graph --graph-steps 3 --graph-alpha 0.85
    python -m src.run_heuristics --model pagerank --graph-damping 0.90
    python -m src.run_heuristics --model ppr --graph-damping 0.85 --graph-max-iter 50

Evaluation protocol:
    # Tune/select on val
    python -m src.run_heuristics --model bayesian --split val --evaluator basic
    # Final report on test (once)
    python -m src.run_heuristics --model bayesian --split test --evaluator basic
    # Extended diagnostics
    python -m src.run_heuristics --model ppr --evaluator offline --ks 10,20 --mode warm_only
"""

from __future__ import annotations

import argparse
import logging

from data.dataframes import movies, test, train, users, val
from src.eval.eval import evaluate as evaluate_basic
from src.eval.offline_ranking import evaluate as evaluate_offline
from src.models.graph import (
    ItemGraphPropagationRanker,
    PageRankRanker,
    PersonalizedPageRankRanker,
)
from src.models.popularity import (
    BayesianPopularityRanker,
    PopularityRanker,
    RecencyPopularityRanker,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate heuristic recommendation models.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="count",
        choices=["count", "bayesian", "recency", "item_graph", "pagerank", "ppr"],
        help=(
            "Heuristic model variant: popularity (count/bayesian/recency) "
            "or graph (item_graph/pagerank/ppr)."
        ),
    )
    parser.add_argument(
        "--evaluator",
        type=str,
        default="basic",
        choices=["basic", "offline"],
        help="Evaluation backend: basic (eval.py) or offline (offline_ranking.py).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Evaluation split; model is always fit on train.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Top-k cutoff for basic evaluator.",
    )
    parser.add_argument(
        "--ks",
        type=str,
        default="10,20",
        help="Comma-separated cutoffs for offline evaluator (e.g., 5,10,20).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=4.0,
        help="Relevance threshold for precision/recall.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "warm_only"],
        help="User subset mode for offline evaluator.",
    )
    parser.add_argument(
        "--bayesian-m",
        type=float,
        default=25.0,
        help="Pseudo-count strength for Bayesian popularity model.",
    )
    parser.add_argument(
        "--half-life-days",
        type=float,
        default=30.0,
        help="Half-life in days for recency popularity model.",
    )
    parser.add_argument(
        "--graph-threshold",
        type=float,
        default=4.0,
        help="Positive-rating threshold for graph edge construction and user seeds.",
    )
    parser.add_argument(
        "--graph-use-rating-weights",
        action="store_true",
        help="Use rating magnitudes as edge weights when building graph.",
    )
    parser.add_argument(
        "--graph-alpha",
        type=float,
        default=0.85,
        help="Propagation alpha for item_graph model.",
    )
    parser.add_argument(
        "--graph-steps",
        type=int,
        default=2,
        help="Number of propagation steps for item_graph model.",
    )
    parser.add_argument(
        "--graph-damping",
        type=float,
        default=0.85,
        help="Damping factor for pagerank and ppr models.",
    )
    parser.add_argument(
        "--graph-max-iter",
        type=int,
        default=100,
        help="Maximum iterations for pagerank and ppr models.",
    )
    parser.add_argument(
        "--graph-tol",
        type=float,
        default=1e-8,
        help="Convergence tolerance for pagerank and ppr models.",
    )

    return parser.parse_args()


def build_model(args: argparse.Namespace):
    if args.model == "count":
        return PopularityRanker()
    if args.model == "bayesian":
        return BayesianPopularityRanker(bayesian_m=args.bayesian_m)
    if args.model == "recency":
        return RecencyPopularityRanker(half_life_days=args.half_life_days)
    if args.model == "item_graph":
        return ItemGraphPropagationRanker(
            relevance_threshold=args.graph_threshold,
            use_rating_weights=args.graph_use_rating_weights,
            alpha=args.graph_alpha,
            n_steps=args.graph_steps,
        )
    if args.model == "pagerank":
        return PageRankRanker(
            relevance_threshold=args.graph_threshold,
            use_rating_weights=args.graph_use_rating_weights,
            damping=args.graph_damping,
            max_iter=args.graph_max_iter,
            tol=args.graph_tol,
        )
    if args.model == "ppr":
        return PersonalizedPageRankRanker(
            relevance_threshold=args.graph_threshold,
            use_rating_weights=args.graph_use_rating_weights,
            damping=args.graph_damping,
            max_iter=args.graph_max_iter,
            tol=max(args.graph_tol, 1e-12),
        )
    raise ValueError(f"Unknown model: {args.model}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    model = build_model(args)
    eval_ratings = val if args.split == "val" else test

    logger.info("Fitting %s on train split...", model.__class__.__name__)
    model.fit(train, users=users, movies=movies)

    if args.evaluator == "basic":
        logger.info("Evaluating with basic evaluator on %s split...", args.split)
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
        logger.info("Model: %s", model.__class__.__name__)
        logger.info("Split: %s", args.split)
        logger.info("NDCG@%d:      %.5f", args.k, metrics.ndcg)
        logger.info("Precision@%d: %.5f", args.k, metrics.precision)
        logger.info("Recall@%d:    %.5f", args.k, metrics.recall)
        return

    ks = [int(v.strip()) for v in args.ks.split(",") if v.strip()]
    logger.info("Evaluating with offline evaluator on %s split...", args.split)
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
    logger.info("Model: %s", model.__class__.__name__)
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
            "k=%d: ndcg=%.5f precision=%.5f recall=%.5f mrr=%.5f",
            k,
            m.ndcg,
            m.precision,
            m.recall,
            m.mrr,
        )


if __name__ == "__main__":
    main()
