# Wide & Deep Recommender

This document describes our Wide & Deep implementation for top-K recommendation on MovieLens.

Important directories:
1. Implementation: `src/models/wide_deep.py` as `WideAndDeepRecommender`
2. Launch script: `src/run_wide_deep.py` for CLI training/evaluation
3. Experiment pipeline: `experiments/WideDeep/WideDeep.ipynb`
4. Experiment report: `experiments/WideDeep/results.md`

## 1. Goal

Wide & Deep combines:
- a **wide** linear component to memorize frequent feature interactions,
- a **deep** neural component to generalize to sparse/unseen combinations.

In this project, we use it as an implicit-feedback ranker:
- positives: interactions with `Rating >= threshold` (default `4.0`),
- negatives: sampled unseen items per user,
- ranking by predicted logits.

This is intended to complement MF/BPR:
- MF/BPR mainly model user-item latent interactions,
- Wide & Deep additionally injects explicit side features and linear memorization terms.

## 2. Architecture

Model core (`_WideAndDeepNet`):

### 2.1 Wide part (linear memorization)

Logit terms:
- global bias
- user bias (`Embedding(n_users, 1)`)
- item bias (`Embedding(n_items, 1)`)
- demographic biases (`gender`, `age`, `occupation`)
- genre linear term (`Linear(n_genres -> 1)`)

### 2.2 Deep part (nonlinear generalization)

Inputs:
- user embedding (`Embedding(n_users, d)`)
- item embedding (`Embedding(n_items, d)`)
- demographic embeddings (`gender`, `age`, `occupation`)
- projected item genre multi-hot (`Linear(n_genres -> genre_embedding_dim)`)

These are concatenated and fed into an MLP:
- `Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear(1)`

### 2.3 Final score

`final_logit = wide_logit + deep_logit`

Ranking during inference is done by sorting candidate items by this score.

## 3. Training Objective

Training uses binary classification on sampled pairs:
- label `1` for positive `(u, i+)`,
- label `0` for sampled negative `(u, j-)`.

Loss:
- `BCEWithLogitsLoss(logit, label)`

Optimization:
- `AdamW`
- optional gradient clipping (`gradient_clip_norm`, default `5.0`)
- stability guards: fail fast on non-finite logits/loss

## 4. Feature Construction

From `users` table:
- `Gender`, `Age`, `Occupation`
- categorical mappings are built once (`*_to_idx`)
- per-user index arrays are cached for fast batch assembly

From `movies` table:
- supports `MovieID` or `movie_id`
- parses `Genres`/`genres` as multi-hot vectors
- fallback to a single dummy genre feature if genre column is absent

Positives:
- ratings filtered by `Rating >= threshold`
- fallback to all interactions if threshold yields empty set

Negatives:
- sampled uniformly from item catalog
- rejection sampling excludes train-seen items for that user

## 5. Inference Behavior

`predict(users, ratings, movies, k)`:
- scores all catalog candidates per user in batches
- masks already-seen items from provided `ratings`
- returns top-K `Rating(movie_id, score)`

Cold-start users:
- if user not in train mapping, fallback to global item popularity counts

## 6. Model Selection and Checkpointing

`fit(...)` supports validation-aware training:
- `val_ratings`: validation split for per-epoch evaluation
- `eval_ks`: Ks used in validation monitor
- `monitor_k`: metric key for best-epoch selection (default `10`)
- `eval_mode`: `"all"` or `"warm_only"`
- `early_stopping_patience`: optional early stop

Best-model handling:
- tracks `best_epoch_`, `best_val_ndcg_`, `val_history_`
- `restore_best_weights=True` restores best epoch at end
- optional checkpoint persistence:
  - `save_checkpoint(path)`
  - `load_checkpoint(path)`

Checkpoint payload includes:
- model state dict
- mappings and cached feature arrays
- training/validation history stats

## 7. Protocol Alignment

This implementation is designed to align with the project’s standard offline protocol:
- global temporal split from `data.dataframes` (`train/val/test`)
- ranking metrics from `src.eval.offline_ranking`
- relevance threshold controlled by `threshold` (default `4.0`)
- evaluation at `K in {10, 20}` by default

Notebook additionally supports per-user temporal split experiments for Two-Tower comparability.