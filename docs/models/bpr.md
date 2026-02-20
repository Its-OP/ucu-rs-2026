# BPR (Bayesian Personalized Ranking)

This document describes our implementation of Bayesian Personalized Ranking (BPR-OPT) for learning-to-rank on implicit feedback derived from MovieLens ratings. 

Important directories:
1. Impementation: `src/models/bpr.py` as `BPRRecommender`
2. Launch script: `src/run_bpr.py` for full CLI
3. Experiment setup&results: `experiments/BPR` folder


## 1. Goal

Instead of predicting absolute ratings, BPR optimizes pairwise ranking: for a user $u$, items the user liked should be ranked higher than items treated as negatives.
In implicit recommendation settings, we often do not trust missing interactions as explicit negatives, 
but BPR becomes a natural fit when the goal is top-$K$ ranking quality, rather than rating regression (RMSE)

## 2. Scoring function

Matrix factorization scoring function with an item bias:

$$
x_{ui} = b_i + \mathbf{p}_u^\top \mathbf{q}_i
$$

- $\mathbf{p}_u \in \mathbb{R}^d$ is the user latent vector
- $\mathbf{q}_i \in \mathbb{R}^d$ is the item latent vector
- $b_i$ is an item bias
- $d$ is `n_factors`

Users and items are mapped to contiguous indices (`user_to_idx`, `item_to_idx`) for array-based training and inference.


## 3. Training objective 

### Pairwise objective

BPR samples triplets $(u, i, j)$, where $i$ is a **positive** item for user $u$, $j$ is a **negative** item sampled from a chosen negative pool.

The goal is to make $x_{ui} > x_{uj}$. Define $\Delta_{uij} = x_{ui} - x_{uj}$. BPR maximizes $\sum_{(u,i,j)} \ln \sigma(\Delta_{uij}) - \lambda \|\Theta\|^2$.

Equivalently, we minimize the **BPR loss**:

$$
\mathcal{L}_{uij} = -\ln \sigma(\Delta_{uij}) + \lambda \|\Theta\|^2
$$

- $\sigma$ is the sigmoid function
- $\Theta$ includes user/item factors and biases
- $\lambda$ is the L2 regularization strength (`regularization`)

Optimization is done with **manual SGD updates** (no external trainer).


## 4. Constructing positives

BPR is trained on implicit positives derived from ratings: `Rating >= threshold` (default `4.0`).

Internally we store, per user index $u$:
- `_user_positive_idx[u]`: array of positive item indices (sample source for $i$)
- `_users_with_positives`: the list of users that can be trained

## 5. Negative sampling

This implementation makes this choice explicit with two switches:

### 5.1 Negative sampling distribution (`negative_sampling`)

We support:
- `uniform`: sample items uniformly from the catalog
- `popularity`: sample items with probability $p(j) \propto \text{count}(j)^\alpha$, where counts are computed from **training interactions only**, and $\alpha$ is `popularity_alpha` (default `0.75`). This popularity-biased sampling tends to produce "harder" negatives than pure uniform sampling.

Negative sampling uses **rejection sampling**: draw a candidate item and resample if it falls into the blocked set. A guard is included to skip rare degenerate cases where a user has seen almost all items.

### 5.2 Negative pool definition (`negative_pool`)

We support:
- `unseen` (recommended for implicit BPR): sample negatives from items **unseen by the user in the training split** and block all items in the user’s training history.
This matches the classical "observed vs unobserved" BPR setup.
- `non_positive`: sample negatives from items that are **not in the positive set**. This treats low-rated items as potential negatives (a hybrid that partially uses explicit signal).
Can be reasonable when low ratings should behave like explicit dislikes, but it changes the interpretation of negatives.

## 6. Optimization procedure

### Sampling schedule

Per epoch, we sample `n_samples` triplets:
- If `n_samples_per_epoch` is set, we use that value
- Otherwise, we use $n\_samples = \max(|\text{ratings}|,\ 10 \cdot |\text{users with positives}|)$
- For each triplet:
  - sample user $u$, positive $i$, negative $j$
  - compute $\Delta_{uij} = x_{ui} - x_{uj}$
  - compute gradient factor $g = \sigma(-\Delta_{uij}) = 1 - \sigma(\Delta_{uij})$
  - apply SGD updates for $\mathbf{p}_u, \mathbf{q}_i, \mathbf{q}_j, b_i, b_j$ with L2 regularization

We store `loss_history_`: average per-sample loss (ranking loss + regularization) per epoch. This is used to study convergence behavior.

## 7. Inference

`predict(users, ratings, movies, k)` returns top-K recommendations for each user as a list of `Rating(movie_id, score)`.
Candidates are items that exist in the provided `movies` table (full catalog), intersected with items known by `item_to_idx`.

For **warm users** $\text{scores} = b + Q \cdot \mathbf{p}_u$, where $Q$ is the matrix of candidate item factors and $b$ is candidate item bias.
We exclude items already interacted with in the provided `ratings` by setting their scores to $-\infty$. This prevents inflating offline metrics by recommending items the user has already seen.

If a user is not present in the trained user mapping (absent from the training split), so called **cold-start**, we cannot compute a personalized score. 
We therefore fall back to a **global popularity prior** computed from training interactions: `scores = global_item_counts[candidates]`.
Seen-item masking is still applied if the user has interactions in the provided `ratings`.


## 8. Hyperparameters

Key parameters exposed via `src/run_bpr.py` and `BPRRecommender`:

- `n_factors`: embedding dimension \(d\)
- `n_epochs`: epochs
- `lr`: learning rate
- `regularization`: L2 coefficient \(\lambda\)
- `n_samples_per_epoch`: training triplets per epoch (useful for fast tuning)
- `threshold`: rating threshold defining positives
- `negative_sampling`: `uniform` or `popularity`
- `negative_pool`: `unseen` or `non_positive`
- `popularity_alpha`: exponent \(\alpha\) for popularity-based sampling
- `random_state`: RNG seed