# Two-Stage Hybrid Recommender

This document describes the design, implementation, and evaluation of the two-stage hybrid recommender that combines BPR collaborative filtering, content-based retrieval, and a LambdaMART learning-to-rank reranker.

Important directories:
1. Implementation: `src/models/hybrid_two_stage.py` as `TwoStageHybridRecommender`
2. Launch script: `src/run_hybrid_two_stage.py` for full CLI
3. Experiment setup & results: `experiments/HybridTwoStage/` folder


## 1. Motivation and Design Rationale

### Why hybrid at all?

BPR and the content-based model each cover a different slice of the recommendation problem:

| Failure mode | BPR (CF) | Content-based (CB) |
|---|---|---|
| Cold users (few ratings) | Falls back to popularity | Can still build a profile from a handful of liked items |
| Tail items (rarely rated) | Scores approach zero — item rarely appears in top-K | Retrieves by semantic similarity regardless of popularity |
| Diverse taste users | Strong if variety is reflected in the interaction matrix | Can surface genre-diverse candidates missed by CF |
| Warm users, mainstream items | Strong | May suggest semantically similar but already-known films |

A hybrid that retrieves from both sources and jointly re-ranks the merged pool addresses these complementary failure modes simultaneously.

### Why a two-stage retrieve-then-rerank structure?

Scoring every item in the catalog (3 706 movies) per-user with a learned ranker is feasible for a small dataset like MovieLens 1M, but the feature construction (CB cosine similarity, rank statistics) for a full-catalog pass is expensive. Retrieval narrows the pool to a tractable size — up to ~520 candidates — while the reranker operates on a much richer, per-candidate feature vector than either retriever uses alone.

This structure also separates two distinct concerns:
- **Retrieval**: maximize recall — get the right items into the pool
- **Reranking**: maximize precision — put the best items at the top of the list

### Why LambdaMART and not a pointwise model?

The original hybrid used a `GradientBoostingRegressor` trained with a pointwise MSE loss on binary liked/not-liked labels. This had two compounding problems:

1. **Loss–metric mismatch.** Minimising MSE over binary labels is not equivalent to maximising NDCG. The model is optimised to predict a score close to 0 or 1, not to order items within a user's candidate group.
2. **Label coarseness.** A 5★ film and a 4★ film both received label 1, while a 3★ film received 0, discarding the magnitude signal entirely.

`LGBMRanker` with `objective="lambdarank"` directly optimises a NDCG-approximating loss (LambdaGrad) over ranked groups. It takes integer relevance grades (0–3), which preserves the full rating magnitude signal within a group.


## 2. Architecture

```
Train ratings
    │
    ├─► BPR (CF retriever)
    │   64 latent factors, SGD, 20 epochs
    │   → top-400 items per user (BPR score)
    │                                            set union → candidate pool (~520 items/user)
    └─► Content-Based (CB retriever)                              │
        SBERT embeddings + FAISS ANN                              │
        → top-120 items per user (cosine sim)                     ▼
                                                         ┌────────────────┐
                                                         │  Stage 2       │
                                                         │  Reranker      │
                                                         │                │
                                                         │  Option A: RRF │ (no learned model)
                                                         │  Option B:     │
                                                         │  LambdaMART    │ (16 features)
                                                         │  + CF blend    │
                                                         └────────┬───────┘
                                                                  │
                                                               top-k list
```

**Stage 1 — Candidate retrieval** runs independently per retriever; results are merged by set union. An item appearing in both pools carries signals from both sources.

**Stage 2 — Reranking** takes the merged pool for each user and produces the final ranked list. Two strategies are supported and were evaluated:
- **RRF**: Reciprocal Rank Fusion — a weighted combination of CF rank and CB rank, with no learned model
- **LambdaMART**: a trained ranker over 16 features, with an optional post-prediction blend with the BPR score


## 3. Stage 1 — Candidate Retrieval

### 3.1 BPR retriever

The collaborative filtering retriever uses the existing `BPRRecommender` without modification. Its scoring function is:

$$
x_{ui} = b_i + \mathbf{p}_u^\top \mathbf{q}_i
$$

where $\mathbf{p}_u \in \mathbb{R}^{64}$ is the user vector, $\mathbf{q}_i \in \mathbb{R}^{64}$ is the item vector, and $b_i$ is the item bias.

At retrieval time, the hybrid calls BPR with `cf_candidates` (default 400) as the pool size. For cold users BPR falls back to global popularity scores.

Two additional outputs of BPR are used downstream:
- The raw **item bias** $b_i$ is exposed as a reranker feature (`bpr_item_bias`), carrying a global item attractiveness signal absent from the dot-product score.
- The **BPR scores** are also stored and used in the post-prediction blend.

### 3.2 Content-based retriever

The content-based retriever uses the `ContentBasedRecommender` infrastructure. Each movie is embedded offline with SentenceBERT (`all-MiniLM-L6-v2`, 384-dim) using weighted field fusion:

$$
\mathbf{e}_i = \text{normalise}\!\left(\frac{0.3 \cdot \mathbf{e}^{\text{title}} + 0.2 \cdot \mathbf{e}^{\text{genre}} + 0.1 \cdot \mathbf{e}^{\text{year}} + 0.4 \cdot \mathbf{e}^{\text{desc}}}{\sum w_k}\right)
$$

The user profile is a recency-decayed weighted mean of liked-item embeddings:

$$
\mathbf{p}_u = \text{normalise}\!\left(\sum_{r \in \mathcal{L}_u} w_r \cdot \text{decay}^{\text{rank}(r)} \cdot \mathbf{e}_{i_r}\right)
$$

where $w_r$ is a rating weight, $\text{decay} = 1.3$ by default, and $\text{rank}(r)$ is the item's position from the newest end.

With `content_n_neighbors=5`, the profile is enriched by blending it with the 5 most similar user profiles, injecting a light collaborative signal into the content retriever.

FAISS `IndexFlatIP` over L2-normalised vectors provides exact cosine nearest-neighbour search. A wider initial pool (`cb_search_size=240`) is retrieved and then trimmed to `cb_candidates=120`.

### 3.3 Candidate pool union

The CF and CB candidate sets are merged by set union. An item may appear in:
- only the CF pool (strong collaborative signal, weak content signal)
- only the CB pool (strong content signal, absent from CF top-400)
- both pools (agreement between retrievers — the strongest signal)

The binary overlap flags `in_cf` and `in_cb` are included as reranker features. In practice, because the BPR pool is large (400 items), the vast majority of candidates appear in both, making these flags near-degenerate. The interaction term `cf_score_norm × cb_score_norm` is more informative for items where both retrievers agree on relevance.


## 4. Stage 2 — Reranking

### 4.1 RRF (ablation baseline)

Reciprocal Rank Fusion scores each candidate as:

$$
s_{\text{RRF}}(i) = \alpha \cdot \frac{1}{\text{rank}_{\text{CF}}(i) + 1} + (1 - \alpha) \cdot \frac{1}{\text{rank}_{\text{CB}}(i) + 1}
$$

Items absent from a retriever are assigned a rank of $+\infty$. The weight `blend_alpha` (default 0.85) is CF-dominant, reflecting BPR's generally higher precision. RRF requires no training data, no held-out labels, and adds no training time. It serves as the zero-reranker ablation baseline.

### 4.2 LambdaMART ranker

#### Training data construction

Reranker training labels are drawn from a **held-out tail** of each user's training timeline:
- 10% of each user's chronologically ordered training ratings are reserved as reranker labels (`rerank_holdout_frac=0.10`)
- For global/test evaluation, the full validation set is used as labels (no holdout needed, since the test set is the evaluation target)

For each training user, the same CF+CB retrieval pipeline generates a candidate pool (smaller training pools: CF=200, CB=60) and features are built for every candidate. Ratings are then converted to integer relevance grades:

| Rating | Grade |
|---|---|
| < 3★ or unseen | 0 |
| 3★ | 1 |
| 4★ | 2 |
| 5★ | 3 |

Groups with all-zero grades are dropped from training (no positive signal, only noise for LambdaGrad).

#### LambdaMART objective

The ranker optimises:

$$
\mathcal{L} = \sum_{u} \sum_{(i,j) \in \mathcal{P}_u} \lambda_{ij} \cdot \Delta \text{NDCG}_{ij}
$$

where $\lambda_{ij}$ are the LambdaGrad weights that scale the pairwise gradient by the NDCG gain from swapping items $i$ and $j$ in the ranked list. This makes the model directly aware of position-dependent gain, unlike a pointwise MSE loss.

Model configuration: `lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8)`.

All users' feature matrices are vstacked into a single array and scored in one `ranker.predict()` call, then sliced back per-user.

#### Feature set

Sixteen features are constructed per (user, candidate) pair:

| Feature | Group | Description |
|---|---|---|
| `cf_score_norm` | CF retriever | BPR score, min-max normalised within the user's pool |
| `cf_rank_inv` | CF retriever | $1 / (\text{rank}_\text{CF} + 1)$ |
| `cf_rank_frac` | CF retriever | $\text{rank}_\text{CF} / N_\text{CF}$ |
| `bpr_item_bias` | CF retriever | Raw $b_i$ from BPR; global item attractiveness |
| `cb_score_norm` | CB retriever | Cosine similarity to user profile, normalised |
| `cb_rank_inv` | CB retriever | $1 / (\text{rank}_\text{CB} + 1)$ |
| `cb_rank_frac` | CB retriever | $\text{rank}_\text{CB} / N_\text{CB}$ |
| `in_cf` | Overlap | Binary: item in CF pool |
| `in_cb` | Overlap | Binary: item in CB pool |
| `cf_cb_interaction` | Overlap | `cf_score_norm × cb_score_norm` |
| `movie_mean_rating` | Item metadata | Global average rating for this movie |
| `log_movie_rating_count` | Item metadata | $\log(1 + \text{rating count})$ — log-compressed popularity |
| `user_mean_rating` | User context | User's historical mean rating |
| `user_rating_count` | User context | Total interactions by this user |
| `user_liked_count` | User context | Interactions ≥ threshold by this user |
| `is_cold_user` | User context | Binary: fewer than `min_liked` positive interactions |

The `movie_rating_count` is log-compressed because its raw distribution is heavily right-skewed; without compression it dominates tree splits and crowds out personalisation features.

#### Post-prediction CF blend

After the LambdaMART scores are computed, the final score is:

$$
s_{\text{final}}(u, i) = \alpha \cdot \text{minmax}_u(s_{\text{ranker}}) + (1 - \alpha) \cdot \text{minmax}_u(s_{\text{CF}})
$$

where $\text{minmax}_u$ normalises scores within user $u$'s candidate pool to $[0, 1]$. At $\alpha = 1.0$ this is pure LambdaMART; at $\alpha = 0.0$ it collapses to BPR ordering. The optimal value found experimentally is $\alpha = 0.6$.

The blend is needed because the reranker trains on a narrower candidate pool than it sees at inference (200 vs 400 CF candidates), causing a covariate shift in the score distribution. The BPR component acts as an anchor that corrects for this shift.


## 5. Training Protocol

The training protocol differs by evaluation split:

| Split | BPR trains on | CB trains on | Reranker labels |
|---|---|---|---|
| `global/val` | `train` minus 10% holdout | same | 10% per-user tail of `train` |
| `global/test` | full `train` | full `train` | full `val` set |
| `per_user/val` | `user_based_temporal_train` minus 10% holdout | same | 10% per-user tail |

For global/val and per_user/val, the holdout is carved **before** BPR and CB are fitted, so the base retrievers never see the reranker labels. This prevents leakage where the CF model would assign elevated scores to items the reranker is supposed to predict.


## 6. Who Benefits

### Users who benefit most

| User type | Why |
|---|---|
| **Warm users with narrow interaction history** | BPR has strong ranking signal but retrieves from a narrow neighbourhood in latent space. CB adds semantically diverse candidates outside that neighbourhood, and the reranker learns to promote the highest-quality ones. |
| **Users with evolving taste** | The CB profile uses recency decay, so recent preferences shift the user embedding. These users may have CF signal anchored on old preferences; CB candidate enrichment via neighbor profiles compensates. |
| **Mainstream users** | BPR already performs well; the hybrid preserves this via the CF blend ($\alpha = 0.6$) while adding a small recall boost from CB. |

### Users who do NOT benefit much

| User type | Why |
|---|---|
| **Cold users** (< `min_liked` positives) | Both the CB user profile and the reranker features are noisy with very few liked items. BPR falls back to popularity and CB retrieves from an imprecise mean profile. |
| **Heavy users** (thousands of ratings) | BPR already covers most of the relevant catalog in its top-400; CB candidates add little new coverage. |

### Items that benefit most

| Item type | Why |
|---|---|
| **Tail items** (few ratings) | BPR's dot-product gives near-zero scores to rarely rated items. If the item has rich metadata (description, genres), CB can retrieve it for users whose profile is similar. |
| **Genre-niche items** | Users with niche tastes may not have enough CF neighbours in that niche. The SBERT genre embedding surfaces semantically close items independent of interaction density. |
| **New items** (zero training ratings) | BPR cannot score them (not in `item_to_idx`). If an item's metadata is present, CB can retrieve it for relevant users from day one. |

### Items that do NOT benefit

| Item type | Why |
|---|---|
| **Mainstream blockbusters** | Already in BPR top-400 for almost every user; CB retrieval does not add new coverage. The reranker may slightly reorder them but cannot improve recall. |
| **Items with missing metadata** | Without title, genres, or description, the SBERT embedding is a zero vector. Such items are invisible to CB retrieval and contribute no CB features to the reranker. |


## 7. Hyperparameters

Key parameters exposed via `src/run_hybrid_two_stage.py`:

**BPR component**
- `--cf-n-factors`: latent dimension $d$ (default 64)
- `--cf-n-epochs`: training epochs (default 20)
- `--cf-lr`: SGD learning rate (default 0.01)
- `--cf-regularization`: L2 coefficient $\lambda$ (default 0.01)
- `--cf-negative-sampling`: `uniform` or `popularity` (default `uniform`)
- `--cf-negative-pool`: `unseen` or `non_positive` (default `unseen`)
- `--cf-popularity-alpha`: exponent for popularity sampling (default 0.75)

**Content-based component**
- `--content-metric`: similarity function, `pearson` or `cosine` (default `pearson`)
- `--content-recency-decay`: recency decay exponent (default 1.3)
- `--content-n-neighbors`: user profile neighbour enrichment count (default 5)
- `--content-min-liked`: minimum liked items to build a valid CB profile (default 5)
- `--content-min-ratings`: minimum movie ratings to include a movie's CB profile (default 100)

**Candidate pools**
- `--cf-candidates`: CF pool size at inference (default 400)
- `--cb-candidates`: CB pool size at inference (default 120)
- `--cb-search-size`: initial FAISS retrieval size before CB trimming (default 240)
- `--train-cf-candidates`: CF pool size when building reranker training data (default 200)
- `--train-cb-candidates`: CB pool size for reranker training (default 60)
- `--train-cb-search-size`: FAISS search size for reranker training (default 120)

**Reranker**
- `--disable-ranker`: skip reranker, use RRF instead
- `--blend-alpha`: RRF CF-weight $\alpha$ when ranker is disabled (default 0.85)
- `--ranker-cf-blend`: post-prediction CF blend weight $\alpha$ (default 1.0; best found: 0.6)
- `--rerank-holdout-frac`: fraction of each user's training tail reserved for reranker labels (default 0.15)

**General**
- `--threshold`: relevance threshold (default 4.0)
- `--mode`: `all` or `warm_only`
- `--seed`: random seed


## 8. Commands to Reproduce

```bash
# Best known config (global/test, LambdaMART blend=0.6)
python -m src.run_hybrid_two_stage \
  --split-type global --split test --evaluator offline --ks 10,20 \
  --mode all --content-n-neighbors 5 \
  --cf-candidates 400 --cb-candidates 120 \
  --train-cf-candidates 200 --train-cb-candidates 60 \
  --ranker-cf-blend 0.6 --rerank-holdout-frac 0.10 --seed 42

# RRF ablation (no ML reranker, global/test)
python -m src.run_hybrid_two_stage \
  --split-type global --split test --evaluator offline --ks 10,20 \
  --mode all --content-n-neighbors 5 \
  --cf-candidates 400 --cb-candidates 120 \
  --disable-ranker --blend-alpha 0.85 --seed 42

# Per-user validation
python -m src.run_hybrid_two_stage \
  --split-type per_user --split val --evaluator offline --ks 10,20 \
  --mode all --content-n-neighbors 5 \
  --cf-candidates 400 --cb-candidates 120 \
  --train-cf-candidates 200 --train-cb-candidates 60 \
  --ranker-cf-blend 0.6 --rerank-holdout-frac 0.10 --seed 42
```
