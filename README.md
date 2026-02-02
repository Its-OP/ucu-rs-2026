# Team Time Management

Movie recommendation system built on the [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) dataset (~1M ratings, 6,040 users, 3,883 movies). Implements and compares collaborative filtering and content-based approaches.

## Repository Structure

```
ucu-rs-2026/
├── data/
│   ├── datasets/               # MovieLens 1M raw files + enriched data
│   │   ├── movies.dat
│   │   ├── users.dat
│   │   ├── ratings.dat
│   │   ├── movies_enriched.csv # Movies with TMDB descriptions
│   │   └── embeddings.npz     # Pre-computed SentenceBERT embeddings
│   ├── dataframes.py           # Data loading, PCA, train/val splits
│   ├── build_movie_embeddings.py  # SentenceBERT embedding pipeline
│   └── enrich_movies.py        # TMDB API enrichment script
│
├── src/
│   ├── models/
│   │   ├── base.py             # RecommenderModel ABC + Rating dataclass
│   │   ├── content_based.py    # Content-based recommender (FAISS + GBR re-ranker)
│   │   ├── collaborative_filtering.py  # Item-Item CF (cosine/adjusted cosine/pearson)
│   │   ├── als.py              # ALS
│   │   └── func_svd.py         # FunkSVD
│   ├── eval/
│   │   ├── eval.py             # Evaluation harness (NDCG@K, Precision@K, Recall@K)
│   │   └── metrics/            # Individual metric implementations
│   │       ├── ndcg.py
│   │       ├── precision.py
│   │       └── recall.py
│   └── run_content_based.py    # CLI entry point for content-based model
│
├── experiments/
│   ├── ALS/                    # ALS notebook + results
│   ├── CF/                     # Item-Item notebook + results
│   └── FunkSVD/                # FunkSVD notebook + results
│
├── eda/
│   ├── movielens_eda.ipynb     # EDA notebook
│   ├── description.md          # EDA summary
│   └── plots/                  # Figures from EDA
│
├── tests/                      # Unit tests for metrics and evaluation
├── requirements.txt
```

## Setup

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
git clone https://github.com/Its-OP/ucu-rs-2026.git
cd ucu-rs-2026

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Data preparation

The MovieLens 1M files (`movies.dat`, `users.dat`, `ratings.dat`) should be placed in `data/datasets/`. The enriched movie data and embeddings are already included in the repository.

To regenerate from scratch (optional):

```bash
# Fetch TMDB descriptions
python -m data.enrich_movies --api-key <your_api_key> --input data/datasets/movies.dat --output data/datasets/movies_enriched.csv

# Build SentenceBERT embeddings
python -m data.build_movie_embeddings data/datasets/movies_enriched.csv data/datasets/embeddings.npz
```

## Usage

### Content-based model

```bash
# Run with best configuration (defaults)
python -m src.run_content_based

# Run with specific parameters
python -m src.run_content_based \
    --scoring gbr_reranker \
    --metric pearson \
    --recency-decay 1.3 \
    --n-neighbors 12

# See all options
python -m src.run_content_based --help
```

Available scoring strategies: `similarity`, `mean_rating`, `hybrid`, `popular`, `gbr_reranker`

Available similarity metrics: `cosine`, `pearson`

### Collaborative filtering experiments

Experiments are run via Jupyter notebooks in the `experiments/` directory:

```bash
jupyter notebook experiments/ALS/ALS.ipynb
jupyter notebook experiments/CF/Collaborative_Filtering.ipynb
jupyter notebook experiments/FunkSVD/FunkSVD.ipynb
```

### Running tests

```bash
pytest tests/
```

## Models

| Model | Type | Best NDCG@10 | Details |
|-------|------|-------------|---------|
| Content-Based + GBR | Content-based | 0.128 | FAISS retrieval + GradientBoostingRegressor re-ranker |
| FunkSVD | Matrix factorisation | 0.091 | 50 factors, lr=0.01, reg=0.02 |
| ALS | Matrix factorisation | 0.035 | 200 factors, reg=0.1 |
| Item-Item CF | Collaborative filtering | 0.030 | Adjusted cosine, K=20 neighbours |
