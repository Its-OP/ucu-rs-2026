# Wide & Deep Recommender

Important directories:
1. Implementation: `src/models/wide_deep.py` as `WideAndDeepRecommender`
2. Launch script: `src/run_wide_deep.py` for CLI training/evaluation
3. Experiment pipeline: `experiments/WideDeep/WideDeep.ipynb`
4. Experiment report: `experiments/WideDeep/results.md`

Paper reference: [Wide & Deep Learning for Recommender Systems (arXiv:1606.07792)](https://arxiv.org/abs/1606.07792).

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

## 8. Required Discussion

### 8.1 Representational Differences vs MF/BPR

- **MF/BPR core representation**
  - Primarily user-item latent interactions (`p_u^T q_i`) plus simple biases.
  - BPR changes the objective to pairwise ranking but keeps similar latent-factor interaction form.
  - Side information is not naturally first-class in vanilla MF/BPR.

- **Wide & Deep representation**
  - **Wide branch** memorizes frequent linear patterns (user/item/demographic biases + genre linear term).
  - **Deep branch** learns nonlinear feature interactions from user/item embeddings plus side features.
  - Metadata (demographics, genres) is directly included in scoring, not just added as post-hoc heuristics.

- **Implication**
  - Wide & Deep can exploit structured metadata when collaborative signals are sparse.
  - MF/BPR can be more compact and robust for pure collaborative structure, but may underuse available side features.

### 8.2 Optimization and Compute Trade-offs

- **Optimization**
  - Wide & Deep uses sampled BCE on implicit positives vs sampled negatives.
  - This is easy to optimize with AdamW, but sensitive to negative-sampling quality and class balance.
  - Stability safeguards (finite checks, gradient clipping) are important; without them, training can diverge.

- **Training cost**
  - Wide & Deep has higher parameter count than compact MF/BPR due to:
    - multiple embedding tables,
    - side-feature projections,
    - MLP layers.
  - Validation-based best-epoch selection and early stopping improve reliability at some extra compute overhead.

- **Inference cost**
  - Current implementation scores full candidate catalog per user in batches.
  - Exact and simple at MovieLens scale, but less scalable than ANN retrieval for very large item catalogs.

- **Sampling trade-off**
  - Uniform negatives are simple and stable.
  - Harder/popularity negatives may improve ranking discrimination, but usually increase variance and tuning complexity.

### 8.3 Why Performance Improves or Degrades

- **Why it improves**
  - Side features provide extra predictive signal beyond user-item ID interaction.
  - Wide path captures memorization patterns; deep path generalizes to sparse or unseen feature combinations.
  - Best-epoch restoration avoids late-epoch degradation.

- **Why it degrades**
  - BCE with sampled negatives is a surrogate; it does not directly optimize global top-K ranking.
  - Easy/biased negative samples can limit ranking gains.
  - Higher-capacity configurations can overfit frequent patterns if regularization/tuning is insufficient.
  - Metric shifts can be driven by split protocol, not only by model quality.

- **Observed split effect in this project**
  - Global temporal split has much higher skip rate than per-user split, so effective evaluated user populations differ.
  - Seen-item masking interacts with split geometry and changes candidate difficulty.
  - Therefore, metric deltas between protocols should be interpreted as a combination of model behavior and evaluation-population differences.
