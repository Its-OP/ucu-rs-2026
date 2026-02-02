"""
Train and evaluate the content-based recommender.

Usage:
    python -m src.run_content_based
    python -m src.run_content_based --scoring gbr_reranker --metric pearson --n-neighbors 12
    python -m src.run_content_based --help
"""

import argparse
import logging

from data.dataframes import movies_enriched, users, user_based_temporal_train, user_based_temporal_val
from src.eval.eval import evaluate
from src.models.content_based import ContentBasedRecommender


logger = logging.getLogger(__name__)

SCORING_CHOICES = ["similarity", "mean_rating", "hybrid", "popular", "gbr_reranker"]
METRIC_CHOICES = ["cosine", "pearson"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the content-based movie recommender.",
    )

    parser.add_argument(
        "--k", type=int, default=10,
        help="Number of recommendations per user (default: 10)",
    )
    parser.add_argument(
        "--threshold", type=float, default=4.0,
        help="Rating threshold for relevance (default: 4.0)",
    )
    parser.add_argument(
        "--n-candidates", type=int, default=200,
        help="Number of FAISS candidates to retrieve per user (default: 200)",
    )
    parser.add_argument(
        "--scoring", type=str, default="gbr_reranker", choices=SCORING_CHOICES,
        help="Scoring strategy (default: gbr_reranker)",
    )
    parser.add_argument(
        "--metric", type=str, default="pearson", choices=METRIC_CHOICES,
        help="Embedding distance metric (default: pearson)",
    )
    parser.add_argument(
        "--beta", type=float, default=0.9,
        help="Hybrid scoring blend factor (default: 0.9)",
    )
    parser.add_argument(
        "--recency-decay", type=float, default=1.3,
        help="Exponential recency decay strength, 0 disables (default: 1.3)",
    )
    parser.add_argument(
        "--n-neighbors", type=int, default=12,
        help="Number of neighbor profiles to average, 0 disables (default: 12)",
    )
    parser.add_argument(
        "--ranker-candidates", type=int, default=50,
        help="Number of FAISS candidates per user for ranker training (default: 50)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    model = ContentBasedRecommender(
        relevance_threshold=args.threshold,
        scoring=args.scoring,
        metric=args.metric,
        beta=args.beta,
        recency_decay=args.recency_decay,
        n_neighbors=args.n_neighbors,
    )

    logger.info("Loading movie embeddings and building FAISS index...")
    model.load(movies_enriched)

    logger.info("Fitting user profiles on train split...")
    model.fit(user_based_temporal_train)

    if args.scoring == "gbr_reranker":
        logger.info("Training GBR re-ranker...")
        model.train_ranker(user_based_temporal_train, n_candidates=args.ranker_candidates)

    logger.info("Running predictions on train split (sanity check)...")
    train_preds = model.predict(
        users, user_based_temporal_train, movies_enriched,
        k=args.k, n_candidates=args.n_candidates,
    )

    n_nonempty = sum(1 for recs in train_preds.values() if len(recs) > 0)
    logger.info(
        "Predictions produced for %d/%d users (non-empty).",
        n_nonempty, users["UserID"].nunique(),
    )

    logger.info("Evaluating on eval split...")
    metrics = evaluate(
        model=model,
        train_ratings=user_based_temporal_train,
        test_ratings=user_based_temporal_val,
        users=users,
        movies=movies_enriched,
        k=args.k,
        threshold=args.threshold,
    )

    logger.info("=== Evaluation ===")
    logger.info("NDCG@%d:      %.5f", args.k, metrics.ndcg)
    logger.info("Precision@%d: %.5f", args.k, metrics.precision)
    logger.info("Recall@%d:    %.5f", args.k, metrics.recall)


if __name__ == "__main__":
    main()
