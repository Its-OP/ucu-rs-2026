import logging
import pandas as pd

from data.dataframes import movies_enriched, users, train, val
from src.eval.eval import evaluate
from src.models.content_based import ContentBasedRecommender


logger = logging.getLogger(__name__)


def main(
    train_ratings: pd.DataFrame,
    eval_ratings: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    k: int = 10,
    threshold: float = 4.0,
    n_candidates: int = 100,
    alpha: float = 1.0,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    model = ContentBasedRecommender(alpha=alpha, relevance_threshold=threshold)

    logger.info("Loading movie embeddings and building FAISS index...")
    model.load(movies)

    logger.info("Fitting user profiles and regressors on train split...")
    model.fit(train_ratings)

    logger.info("Running predictions on train split (sanity check)...")
    train_preds = model.predict(users, train_ratings, movies, k=k, n_candidates=n_candidates)

    n_nonempty = sum(1 for recs in train_preds.values() if len(recs) > 0)
    logger.info("Predictions produced for %d/%d users (non-empty).", n_nonempty, users["UserID"].nunique())

    logger.info("Evaluating on eval split...")
    metrics = evaluate(
        model=model,
        train_ratings=train_ratings,
        test_ratings=eval_ratings,
        users=users,
        movies=movies,
        k=k,
        threshold=threshold,
    )

    logger.info("Done.")
    print("\n=== Evaluation ===")
    print(f"NDCG@{k}:      {metrics.ndcg:.5f}")
    print(f"Precision@{k}: {metrics.precision:.5f}")
    print(f"Recall@{k}:    {metrics.recall:.5f}")


if __name__ == "__main__":
    main(
        train_ratings=train,
        eval_ratings=val,
        users=users,
        movies=movies_enriched,
        k=10,
        threshold=4.0,
        n_candidates=200,
        alpha=1.0,
    )