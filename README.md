# Team Time Management

Movie recommendation system built on the [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) dataset (~1M ratings, 6,040 users, 3,883 movies).

## Homework parts

| Part | Models covered | Summary report |
|------|---------------|----------------|
| **HW1** | Popularity baseline, Item-Item CF, FunkSVD, ALS, Content-based | [reports/summary_report_1.md](reports/summary_report_1.md) |
| **HW2** | Heuristics, BPR, Two-stage hybrid, Two-tower, Wide & Deep, A/B test, Bandits | [reports/summary_report_2.md](reports/summary_report_2.md) |

Evaluation strategy and metric definitions are in [reports/evaluation/evaluation_strategy.md](reports/evaluation/evaluation_strategy.md).

---

## Repository structure

```
ucu-rs-2026/
├── data/
│   ├── datasets/               # MovieLens 1M raw files + enriched data + embeddings
│   ├── dataframes.py           # Data loading, train/val/test splits
│   ├── build_movie_embeddings.py
│   └── enrich_movies.py
│
├── src/
│   ├── models/                 # Model implementations
│   ├── eval/                   # Evaluation policy (eval.py, offline_ranking.py, metrics/)
│   ├── run_*.py                # CLI entry points (one per model, see below)
│   └── train_two_tower_transformer.py
│
├── experiments/                # Notebooks + results per model (see below)
├── runs/                       # Timestamped reports for two-tower and bandit runs
├── reports/
│   ├── models/                 # Per-model design docs (and results for some models)
│   ├── evaluation/             # Evaluation strategy, A/B test plan
│   ├── summary_report_1.md
│   └── summary_report_2.md
├── eda/                        # EDA notebook and plots
└── tests/                      # Unit tests for metrics
```

---

## Model index

The table below shows where to find each model's documentation, implementation, run entry point,
and results. Three patterns to keep in mind:

- **ALS, Item-Item CF, FunkSVD** — run exclusively via Jupyter notebooks; there is no CLI run
  script for these models.
- **Most models** (Heuristics, BPR, Two-stage hybrid, Wide & Deep) — have a CLI run script in
  `src/` and a dedicated notebook + `results.md` inside `experiments/<ModelName>/`.
- **Two-tower and Bandits** — have a CLI run script but their results
  are in timestamped run reports under `experiments/` folder. Content-based follows the same pattern but results are in the model doc.

| Model | Design doc | Implementation | Run entry point | Results |
|-------|-----------|----------------|-----------------|---------|
| Popularity baseline | — | `src/models/popularity/ranker.py` | `src/run_popularity.py` | `experiments/Heuristics/results.md` |
| Item-Item CF | — | `src/models/collaborative_filtering.py` | `experiments/CF/Collaborative_Filtering.ipynb` _(notebook only)_ | `experiments/CF/results.md` |
| FunkSVD | — | `src/models/func_svd.py` | `experiments/FunkSVD/FunkSVD.ipynb` _(notebook only)_ | `experiments/FunkSVD/results.md` |
| ALS | — | `src/models/als.py` | `experiments/ALS/ALS.ipynb` _(notebook only)_ | `experiments/ALS/results.md` |
| Content-based | `reports/models/content_based.md` _(includes results)_ | `src/models/content_based.py` | `src/run_content_based.py` | see model doc |
| Heuristics | — | `src/models/heuristic_base.py`, `src/models/graph/rankers.py` | `src/run_heuristics.py` | `experiments/Heuristics/results.md` |
| BPR | `reports/models/bpr.md` | `src/models/bpr.py` | `src/run_bpr.py` | `experiments/BPR/results.md` |
| Two-stage hybrid | `reports/models/hybrid_two_stage.md` | `src/models/hybrid_two_stage.py` | `src/run_hybrid_two_stage.py` | `experiments/HybridTwoStage/results.md` |
| Two-tower transformer | `reports/models/DL_recommenders/two_towers.md` _(includes results)_ | `src/models/ANN/two_towers/` | `src/train_two_tower_transformer.py` | `experiments/two_tower_transformer_*/report.md` |
| Wide & Deep | `reports/models/DL_recommenders/wide_deep.md` | `src/models/wide_deep.py` | `src/run_wide_deep.py` | `experiments/WideDeep/results.md` |
| Bandits | `reports/models/bandits.md` _(includes results)_ | `src/models/bandit/` | `src/run_bandit.py` | `experiments/bandit_*/report.md` |

---

## Results leaderboards

### Global temporal split (75 / 12.5 / 12.5)

Best configuration per model family, ordered by NDCG@10.
HW1 models (CF, ALS, FunkSVD) report **validation** metrics — the final test set was not
separately evaluated in HW1 and `eval.py` did not compute MRR or Precision consistently.
All HW2 models report **test** metrics.

| Model | NDCG@10 | Precision@10 | Recall@10 | MRR@10 | Eval set |
|---|---:|---:|---:|---:|---|
| Item-Item CF (adj cosine, k=20) | 0.030 | 0.029 | 0.006 | — | val |
| ALS (200 factors, reg=0.1) | 0.035 | 0.030 | 0.005 | — | val |
| FunkSVD (50 factors, lr=0.01) | 0.091 | 0.083 | 0.021 | — | val |
| popularity\_count | 0.217 | 0.193 | 0.042 | 0.351 | test |
| Wide & Deep (wd\_e64\_h128x64\_neg2) | 0.221 | 0.199 | 0.043 | 0.347 | test |
| item\_graph (α=0.85, steps=1) | 0.235 | 0.213 | 0.049 | 0.368 | test |
| BPR (uniform, unseen) | 0.239 | 0.212 | 0.050 | 0.385 | test |
| **Hybrid LambdaMART (blend=0.6)** | **0.248** | **0.217** | **0.055** | **0.393** | **test** |

Precision for Hybrid LambdaMART on global test was not reported in the CLI sweep.
Val numbers for HW1 models are not comparable to HW2 test numbers — the global val window
evaluates a different (higher-engagement) user subset (e.g. BPR val = 0.296 vs test = 0.239).

### Per-user temporal split (75 / 25, validation only)

All results are on the per-user validation set. No BPR standalone result exists on this
protocol; it appears only as the retrieval backbone inside the hybrid.

| Model | NDCG@10 | Precision@10 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|
| Content-based (Pearson + recency) | 0.053 | 0.039 | 0.020 | — |
| Content-based GBR re-ranker (full pipeline) | 0.128 | 0.110 | 0.052 | — |
| Hybrid LambdaMART (blend=0.6) | 0.126 | 0.108 | 0.050 | 0.231 |
| Wide & Deep (wd\_e96\_h192x96\_neg3) | 0.128 | 0.109 | 0.049 | 0.239 |
| Two-tower transformer (epoch 15) | 0.137 | 0.100 | 0.063 | 0.217 |
| **Hybrid RRF no-ranker (α=0.85)** | **0.143** | **0.123** | **0.058** | **0.265** |

---

## Setup

```bash
git clone https://github.com/Its-OP/ucu-rs-2026.git
cd ucu-rs-2026

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

The MovieLens 1M files and pre-computed embeddings are already included in `data/datasets/`.
To regenerate embeddings from scratch (optional):

```bash
python -m data.enrich_movies --api-key <your_tmdb_key> \
    --input data/datasets/movies.dat --output data/datasets/movies_enriched.csv

python -m data.build_movie_embeddings \
    data/datasets/movies_enriched.csv data/datasets/embeddings.npz
```

---

## Usage

All CLI-based models follow the same pattern — pass `--help` for the full option list:

```bash
python -m src.run_bpr --help
python -m src.run_heuristics --help
python -m src.run_content_based --help
python -m src.run_hybrid_two_stage --help
python -m src.run_wide_deep --help
python -m src.run_bandit --help
python -m src.train_two_tower_transformer --help
```

Common flags shared across most run scripts:

```bash
# evaluation split and protocol
--split-type global   # global temporal split (default)
--split-type per_user # per-user temporal split
--split val           # evaluate on validation set
--split test          # evaluate on held-out test set

# evaluation cutoffs and mode
--ks 10,20
--mode all            # all users (default)
--mode warm_only      # users with at least one training positive
```

Notebook-only models (ALS, Item-Item CF, FunkSVD):

```bash
jupyter notebook experiments/ALS/ALS.ipynb
jupyter notebook experiments/CF/Collaborative_Filtering.ipynb
jupyter notebook experiments/FunkSVD/FunkSVD.ipynb
```

Two-tower and bandit runs write a timestamped Markdown report to `runs/` (top-level, not inside `experiments/`) automatically.

---

## Tests

```bash
pytest tests/
```
