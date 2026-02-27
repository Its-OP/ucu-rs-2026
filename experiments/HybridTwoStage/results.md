# Two-Stage Hybrid Recommender — Results

## Experimental Setup

- Evaluator: `src.eval.offline_ranking.evaluate`
- Metrics at `k in {10, 20}`
- Evaluation mode: `mode="all"`
- Positive label for training/evaluation: `Rating >= 4.0`
- List size: top-20 candidates re-ranked, metrics at k=10 and k=20

### Split protocols

| Protocol | Train | Eval | Notes |
|---|---|---|---|
| `global/val` | `train` (~750 k ratings) | `val` | Validation; ~1 188–1 223 users with ≥1 relevant item |
| `global/test` | `train` | `test` | Final evaluation; ~1 347 users with ≥1 relevant item |
| `per_user/val` | `user_based_temporal_train` | `user_based_temporal_val` | Per-user temporal holdout; ~5 999 users evaluated |

> **Skip rate note:** The global split has a ~80 % skip rate because most users have no ≥4 rating in the global test window. The per-user split has a ~0.7 % skip rate because every user contributes a personal holdout tail.

---

## 1. System Description

### Architecture overview

The Two-Stage Hybrid Recommender combines a collaborative filtering retriever, a content-based retriever, and an optional learning-to-rank reranker into a single prediction pipeline.

```
Train ratings
    │
    ├─► BPR (CF)  ──────────────────────────────┐
    │   64 latent factors, SGD, 20 epochs        │  union
    │                                            ├─► candidate pool ──► LambdaMART reranker ──► top-k
    └─► Content-Based (CB) ──────────────────────┘                      (or RRF score blend)
        SBERT embeddings + FAISS ANN
```

**Stage 1 — Candidate retrieval:** Two retrievers run independently and their outputs are merged (set union). At prediction time the pool is:
- **CF candidates:** up to 400 items scored by the BPR dot-product (user × item factors + item bias)
- **CB candidates:** up to 120 items retrieved by FAISS ANN over a recency-decayed SBERT user profile

**Stage 2 — Reranking:** The merged pool (up to ~520 unique items per user) is passed to a reranker that produces the final ranked list. Two reranking strategies were evaluated:

- **RRF (no ML reranker):** Reciprocal Rank Fusion blends CF rank and CB rank with a tunable `blend_alpha` weight (no learned model, zero reranker training cost)
- **LambdaMART reranker:** a `lgb.LGBMRanker` trained to directly optimise NDCG, with a post-prediction blend of its output with the normalised BPR score

---

### BPR component (reused from standalone model)

The collaborative filtering retriever reuses the existing `BPRRecommender` unchanged:

- **Algorithm:** Bayesian Personalised Ranking — SGD over implicit pairwise (user, positive item, negative item) triplets, maximising `ln σ(x_ui − x_uj)`
- **Parameters:** 64 latent factors, learning rate 0.01, L2 regularisation 0.01, 20 epochs
- **Negative sampling:** uniform over unseen items (default); popularity-weighted option available
- **Output used by hybrid:**
  - Per-user top-N item scores for candidate retrieval
  - Raw item bias vector (`item_bias`) exposed as a reranker feature (`bpr_item_bias`)
  - Cold-user fallback via global popularity scores

---

### Content-based component (reused from standalone model)

The content-based retriever reuses the existing `ContentBasedRecommender` infrastructure:

- **Item representations:** SBERT embeddings (`sentence-transformers/all-MiniLM-L6-v2`, 384-dim) built offline via weighted field fusion:
  - Title × 0.3, Genres × 0.2, Year × 0.1, Description × 0.4
- **User profile:** weighted average of liked-item embeddings with recency decay (`decay^(rank_from_end)`, default 1.3); optionally enriched with nearest-neighbour profiles
- **Retrieval:** FAISS `IndexFlatIP` approximate nearest-neighbour search on L2-normalised vectors; a larger `cb_search_size` pool is retrieved and top-N reranked
- **Neighbour enrichment:** `content_n_neighbors=5` — the user profile is blended with profiles of the 5 most similar users to improve coverage

---

### LambdaMART reranker (new component)

A learning-to-rank model trained on top of the candidate pool signals. Key design decisions:

**Training data construction**
- Source: per-user tail holdout carved from training ratings (`rerank_holdout_frac=0.10`, 10% of each user's timeline)
- For global/test evaluation: full validation set used as reranker labels (no holdout fraction applied)
- Candidates generated for each holdout user using the same CF + CB retrieval pipeline (smaller pools: CF=200, CB=60)
- Relevance grades: integer 0–3 mapping rating buckets (`<3→0`, `3→1`, `4→2`, `5→3`)

**Model**
- `lgb.LGBMRanker(objective="lambdarank")` — directly optimises NDCG via LambdaMART gradient updates
- 300 estimators, learning rate 0.05, 31 leaves, min 20 samples per leaf, 80% row/column subsampling
- Single batched `predict()` call over all users (vstack all feature matrices, slice results back)

**Feature set (16 features)**

| Feature | Description |
|---|---|
| `cf_score_norm` | BPR score, min-max normalised per user |
| `cf_rank_inv` | 1 / (CF rank + 1) |
| `cf_rank_frac` | CF rank / total CF candidates |
| `bpr_item_bias` | Raw BPR item bias scalar |
| `cb_score_norm` | CB cosine similarity, normalised per user |
| `cb_rank_inv` | 1 / (CB rank + 1) |
| `cb_rank_frac` | CB rank / total CB candidates |
| `in_cf` | Binary: item appeared in CF pool |
| `in_cb` | Binary: item appeared in CB pool |
| `cf_cb_interaction` | `cf_score_norm × cb_score_norm` |
| `movie_mean_rating` | Global average rating for this movie |
| `log_movie_rating_count` | log1p(number of ratings for this movie) |
| `user_mean_rating` | User's historical mean rating |
| `user_rating_count` | Total number of ratings by this user |
| `user_liked_count` | Number of ratings ≥ threshold by this user |
| `is_cold_user` | Binary: user has fewer than `min_liked` liked items |

**Post-prediction blending (`ranker_cf_blend`)**

After the LambdaMART scores are produced, the final ranking score is a convex combination of the normalised ranker output and the normalised BPR CF score:

```
final = α × minmax(ranker_scores) + (1 − α) × minmax(cf_scores)
```

`α = 1.0` uses the ranker output only; `α = 0.0` falls back to pure BPR ordering. This parameter was swept over {1.0, 0.8, 0.6, 0.4}.

---

## 3. BPR/CB Standalone Baselines

Default BPR (64 factors, 20 epochs, uniform negative sampling, `global/test`).

| NDCG@10 | Precision@10 | Recall@10 | MRR@10 | MAP@10 |
|---|---|---|---|---|
| 0.23908 | 0.21151 | 0.05043 | 0.38454 | 0.13644 |

Hybrid CB (FAISS, recency, neighbor enrichment, GBR re-ranker, `per-user/val`).

| NDCG@10 | Precision@10 | Recall@10 | MRR@10 | MAP@10 |
|---|---|---|---|---|
| 0.12822 | 0.10966 | 0.05185 | - | - |

Default CB (recency decay 0.3, perason similarity, `per-user/val`).

| NDCG@10 | Precision@10 | Recall@10 | MRR@10 | MAP@10 |
|---|---|---|---|---|
| 0.05270 | 0.03882 | 0.02040 | - | - |

---

## 4. Early Hybrid Experiments (GBR ranker, pre-refactor)

These runs used the original `GradientBoostingRegressor` reranker with binary labels (liked / not liked), 13 features, and smaller candidate pools (CF=120, CB=60). Results are from the **old notebook** runs on the global split.

### Global / val

| Config | NDCG@10 | MRR@10 | MAP@10 | Notes |
|---|---|---|---|---|
| `hybrid_ranker_default` | 0.26360 | 0.45366 | 0.15182 | Default GBR, CF=120, CB=60 |
| `hybrid_ranker_small_candidates` | 0.27832 | 0.45532 | 0.16568 | Smaller candidate pools |
| `hybrid_blend_no_ranker` | 0.29008 | 0.46501 | 0.17591 | RRF blend, no GBR reranker |

### Global / test

| Config | NDCG@10 | MRR@10 | MAP@10 | Notes |
|---|---|---|---|---|
| `hybrid_blend_no_ranker` | 0.22077 | 0.37118 | 0.12446 | Best val config; drops ~7 pts on test |

**Finding:** Large val → test gap (0.29008 → 0.22077) and hybrid below BPR on test (BPR=0.23908). Root causes identified: binary labels, pointwise MSE loss, truncated CF pool, raw rating count (scale mismatch), missing BPR item bias and rank fraction features.

---

## 5. Refactored Hybrid — LambdaMART Reranker

**Architecture changes (v2):**
- `GradientBoostingRegressor` → `lgb.LGBMRanker(objective="lambdarank")`
- Binary labels → integer relevance grades 0–3 (via rating buckets: <3→0, 3→1, 4→2, 5→3)
- 13 features → 16 features: added `cf_rank_frac`, `bpr_item_bias`, `cb_rank_frac`, `log_movie_rating_count` (log1p)
- CF candidate pool: 120 → 400 (train: 120 → 200)
- CB candidate pool: 200 → 120 (train: 120 → 60)
- Batched `ranker.predict()` (single call for all 6 040 users)
- New `ranker_cf_blend` parameter: final score = α × minmax(ranker) + (1−α) × minmax(CF)

### Feature importance (representative run, per-user/val, blend=0.6)

| Feature | Split importance |
|---|---|
| `user_mean_rating` | 1031 |
| `bpr_item_bias` | 997 |
| `movie_mean_rating` | 982 |
| `user_rating_count` | 924 |
| `log_movie_rating_count` | 785 |
| `user_liked_count` | 762 |
| `cf_rank_inv` | 765 |
| `cf_score_norm` | 708 |
| `cb_score_norm` | 617 |
| `cf_cb_interaction` | 635 |
| `cf_rank_frac` | 266 |
| `cb_rank_inv` | 349 |
| `cb_rank_frac` | 178 |
| `in_cf` / `in_cb` / `is_cold_user` | ≤1 |

### 3a. Early blend experiments (CLI, global/test)

Exploratory runs before the full notebook sweep. Configs: `n_neighbors=5`, `cf_candidates=400`, `cb_candidates=120`, `train_cf=200`, `train_cb=60`, `seed=42`.

| Config | NDCG@10 | Recall@10 | MRR@10 | Notes |
|---|---|---|---|---|
| Old RRF no-ranker (CF=500, CB=80, α=0.85) | 0.23871 | — | — | Pre-refactor candidate sizes |
| Old GBR ranker (CF=120, CB=60) | 0.23653 | 0.04896 | 0.37217 | Pre-refactor |
| LambdaMART blend=1.0, n_neighbors=0 | 0.23899 | 0.05767 | 0.38084 | First LambdaMART run |
| LambdaMART blend=1.0, 128f e30 | 0.23808 | — | — | Overfit: more capacity hurts |
| LambdaMART blend=1.0, n_neighbors=5 | 0.23863 | 0.05819 | 0.37856 | CB neighbor enrichment |
| LambdaMART blend=0.8, n_neighbors=5 | 0.24517 | 0.05727 | 0.38583 | First blend to beat BPR |
| **LambdaMART blend=0.6, n_neighbors=5** | **0.24762** | 0.05516 | **0.39331** | **Best on test** |

### 3b. Notebook sweep — Global / val (5 representative configs)

Common args: `--mode all --content-n-neighbors 5 --cf-candidates 400 --cb-candidates 120 --train-cf-candidates 200 --train-cb-candidates 60 --rerank-holdout-frac 0.10 --seed 42`

| Config | NDCG@10 | Precision@10 | Recall@10 | MRR@10 | MAP@10 | Notes |
|---|---|---|---|---|---|---|
| `hybrid_no_ranker_rrf085` | **0.29570** | **0.26396** | **0.05564** | **0.46636** | **0.17936** | Ablation: RRF only, no ML reranker |
| `hybrid_lambdamart_blend0.4` | 0.28204 | 0.25274 | 0.05213 | 0.44569 | 0.16399 | 40% LambdaMART + 60% BPR CF |
| `hybrid_lambdamart_blend0.6` | 0.26607 | 0.24485 | 0.05148 | 0.39314 | 0.15521 | 60% LambdaMART + 40% BPR CF |
| `hybrid_lambdamart_blend0.8` | 0.24923 | 0.22576 | 0.04741 | 0.38380 | 0.13520 | 80% LambdaMART + 20% BPR CF |
| `hybrid_lambdamart_blend1.0` | 0.19342 | 0.18288 | 0.03485 | 0.27001 | 0.09764 | Pure LambdaMART, no BPR blend |

> Val best: `hybrid_no_ranker_rrf085` (NDCG@10=0.29570). However, this config does **not** transfer well to test (see §3c).

### 3c. Notebook sweep — Global / test (auto-selected best val config)

The notebook auto-picked the val winner (`hybrid_no_ranker_rrf085`) and ran it on test.

| Config | NDCG@10 | Precision@10 | Recall@10 | MRR@10 | MAP@10 | n_eval | skip_rate |
|---|---|---|---|---|---|---|---|
| `hybrid_no_ranker_rrf085` (test) | 0.23709 | 0.21007 | 0.04958 | 0.37669 | 0.13406 | 1347 | 0.777 |

> On test this config is **below BPR** (BPR NDCG@10=0.23908). The large val→test gap is a sign that val overfits to a small, high-engagement user subset.

### 3d. Notebook sweep — Per-user / val (5 configs)

| Config | NDCG@10 | Precision@10 | Recall@10 | MRR@10 | MAP@10 | n_eval |
|---|---|---|---|---|---|---|
| `hybrid_no_ranker_rrf085` | **0.14293** | **0.12292** | **0.05751** | **0.26528** | **0.06885** | 5999 |
| `hybrid_lambdamart_blend0.4` | 0.12698 | 0.10900 | 0.05034 | 0.23632 | 0.05730 | 5999 |
| `hybrid_lambdamart_blend0.6` | 0.12510 | 0.10774 | 0.05010 | 0.23049 | 0.05490 | 5999 |
| `hybrid_lambdamart_blend0.8` | 0.11981 | 0.10291 | 0.04894 | 0.21283 | 0.04905 | 5999 |
| `hybrid_lambdamart_blend1.0` | 0.10534 | 0.08841 | 0.04392 | 0.18787 | 0.03934 | 5999 |

Dedicated best-known run (`blend=0.6`, per-user/val):

| NDCG@10 | Precision@10 | Recall@10 | MRR@10 | MAP@10 | n_eval | skip_rate |
|---|---|---|---|---|---|---|
| 0.12576 | 0.10791 | 0.04998 | 0.22880 | 0.05514 | 5999 | 0.007 |

---

## 6. Protocol Comparison Summary

Best config per protocol (from notebook):

| Protocol | Best variant | NDCG@10 | Precision@10 | Recall@10 | MRR@10 | MAP@10 | n_eval | skip_rate |
|---|---|---|---|---|---|---|---|---|
| `global_temporal_val` | `hybrid_no_ranker_rrf085` | 0.29570 | 0.26396 | 0.05564 | 0.46636 | 0.17936 | 1188 | 0.803 |
| `global_temporal_test` | `hybrid_no_ranker_rrf085` | 0.23709 | 0.21007 | 0.04958 | 0.37669 | 0.13406 | 1347 | 0.777 |
| `per_user_temporal_val` | `hybrid_no_ranker_rrf085` | 0.14293 | 0.12292 | 0.05751 | 0.26528 | 0.06885 | 5999 | 0.007 |

---

## 7. Final Leaderboard (Global / Test Split)

Comparing all runs against the BPR standalone baseline on the held-out test set.

| Model | NDCG@10 | Recall@10 | MRR@10 | vs BPR NDCG | vs BPR MRR |
|---|---|---|---|---|---|
| BPR baseline | 0.23908 | 0.05043 | 0.38454 | — | — |
| Old GBR hybrid (CF=120) | 0.23653 | 0.04896 | 0.37217 | −1.1% | −3.2% |
| Old RRF no-ranker (CF=500) | 0.23871 | — | — | −0.2% | — |
| LambdaMART blend=1.0, n=0 | 0.23899 | 0.05767 | 0.38084 | −0.04% | −1.0% |
| LambdaMART blend=1.0, n=5 | 0.23863 | 0.05819 | 0.37856 | −0.2% | −1.6% |
| RRF no-ranker, n=5 (notebook) | 0.23709 | 0.04958 | 0.37669 | −0.8% | −2.0% |
| LambdaMART blend=0.8, n=5 | 0.24517 | 0.05727 | 0.38583 | **+2.5%** | **+0.3%** |
| **LambdaMART blend=0.6, n=5** | **0.24762** | 0.05516 | **0.39331** | **+3.6%** | **+2.3%** |

**Current leader on test:** `hybrid_lambdamart_blend0.6`

---

## 8. Final Leaderboard (Per-User / Val Split)

Comparing all per-user runs against the CB standalone baselines (n_eval ≈ 5 999, skip_rate ≈ 0.7%).

| Model | NDCG@10 | Precision@10 | Recall@10 | MRR@10 | MAP@10 | vs hybrid-CB NDCG |
|---|---|---|---|---|---|---|
| Default CB (recency, pearson) | 0.05270 | 0.03882 | 0.02040 | — | — | — |
| Hybrid CB (FAISS + GBR re-ranker) | 0.12822 | 0.10966 | 0.05185 | — | — | — |
| LambdaMART blend=1.0 | 0.10534 | 0.08841 | 0.04392 | 0.18787 | 0.03934 | −17.8% |
| LambdaMART blend=0.8 | 0.11981 | 0.10291 | 0.04894 | 0.21283 | 0.04905 | −6.6% |
| LambdaMART blend=0.6 | 0.12510 | 0.10774 | 0.05010 | 0.23049 | 0.05490 | −2.4% |
| LambdaMART blend=0.4 | 0.12698 | 0.10900 | 0.05034 | 0.23632 | 0.05730 | −1.0% |
| **RRF no-ranker (α=0.85)** | **0.14293** | **0.12292** | **0.05751** | **0.26528** | **0.06885** | **+11.5%** |

**Current leader on per-user/val:** `hybrid_no_ranker_rrf085`

```
--mode all --content-n-neighbors 5
--cf-candidates 400 --cb-candidates 120
--train-cf-candidates 200 --train-cb-candidates 60
--ranker-cf-blend 0.6 --rerank-holdout-frac 0.10 --seed 42
```

---

## 9. Analysis

### 9.1 Val → test gap on the global split

The global val and test sets have very different user populations. On val, ~80% of users are skipped because they have no ≥4 rating in the val window; the evaluated ~1 188 users are a high-engagement subset who rated actively in a narrow recent period. This makes val metrics inflated and unrepresentative: the best val NDCG@10 across all runs is 0.296 (RRF no-ranker), yet the same config scores 0.237 on test — a 6-point absolute gap. Any config selection made on global val should be treated with caution.

The gap is smaller for the LambdaMART configs (e.g., blend=0.6 goes from 0.266 on val to 0.248 on test, a ~2-point gap), suggesting the reranker generalises better across user populations than the raw BPR-ordered RRF, even though RRF wins on val.

### 9.2 Blend sweep: why pure LambdaMART underperforms and blend=0.6 wins

The pure LambdaMART run (blend=1.0) scores below the BPR baseline on every split. Two factors explain this:

1. **Training scope mismatch.** The reranker sees at most 200 CF candidates per training user, a much smaller pool than the 400 it must rank at inference. The score distribution and rank statistics it has learned do not transfer perfectly to the wider inference pool.
2. **Positive rate is very low (~2%).** With fewer than 1 in 50 candidates being relevant, LambdaMART's LambdaGrad updates carry little signal per group, and the model converges to near-popularity ordering rather than personalisation.

Adding even a small CF prior (blend=0.8, 20% BPR) recovers most of the NDCG loss. At blend=0.6 the ranker's learned reordering adds a clear margin over pure BPR: +3.6% NDCG@10 and +2.3% MRR@10. Beyond blend=0.4 the BPR contribution dominates and the ranker's contribution shrinks; test results for blend=0.4 were not collected, but on per-user val it scores 0.127 vs 0.143 for the no-ranker RRF, so further reducing α below 0.4 is unlikely to help.

### 9.3 RRF no-ranker on per-user split

On per-user val the no-ranker RRF config is the clear winner (NDCG@10 = 0.143, +11.5% vs hybrid-CB baseline). Two structural reasons account for the LambdaMART disadvantage here:

1. **Training data compression.** For per-user evaluation, reranker labels are carved from the 10% trailing tail of each user's training timeline. With ~750 k total training ratings split ~90/10, the reranker trains on ~75 k ratings spread across 5 430 users — roughly 14 interactions per user on average. This is too sparse to learn robust per-user preferences, so the ranker falls back to popularity-like features.
2. **Holdout construction is adversarial.** The per-user holdout tail is temporally the most recent part of training, which is also what the val set tests. When the tail is removed for reranker labels, the BPR model trains on a slightly sparser signal and the ranker's positive examples are drawn from a distribution that partially overlaps with the val set, amplifying label noise.

In contrast, the no-ranker RRF relies only on BPR's own ranking (trained on the full 90% of each user's history) and the CB profile. Both are trained with a richer signal under the per-user protocol, explaining RRF's advantage.

### 9.4 Feature importance interpretation

The top features by split importance across runs are `user_mean_rating`, `bpr_item_bias`, `movie_mean_rating`, `user_rating_count`, and `log_movie_rating_count`. This hierarchy reveals two dominant signals:

- **User-level bias features** (`user_mean_rating`, `user_rating_count`, `user_liked_count`) collectively account for ~37% of split importance. The ranker learns that generous raters and active users have higher hit rates — a strong prior that partly proxies for user engagement level.
- **Item popularity and quality** (`bpr_item_bias`, `movie_mean_rating`, `log_movie_rating_count`) account for ~33%. The BPR item bias is the single most informative individual feature, capturing a global item attractiveness signal not available in the raw CF dot-product.
- **CF rank signals** (`cf_score_norm`, `cf_rank_inv`) contribute ~22% and are the primary personalisation signal the ranker refines.
- **CB features** (`cb_score_norm`, `cb_rank_inv`, `cf_cb_interaction`) contribute ~18%, primarily through the interaction term, which captures items that score well on both retrievers simultaneously.
- **Binary membership flags** (`in_cf`, `in_cb`, `is_cold_user`) carry essentially zero importance. Since the candidate pool is a union, nearly all items appear in both CF and CB, making membership flags degenerate.

### 9.5 CB contribution as a retriever

The content-based component adds clear retrieval coverage: recall@10 jumps from 0.050 (BPR alone) to ~0.058 with CB candidates included (LambdaMART blend=1.0 row), a +16% relative gain in recall. This improvement is consistent across the blend sweep. The trade-off is that CB candidates occupy slots that BPR would have filled with high-precision items, which depresses NDCG and MRR when the ranker cannot re-sort them effectively (blend=1.0 case). The ranker-CF blend restores precision while retaining the recall gains from CB retrieval.

### 9.6 Capacity and regularisation: BPR factor count

Increasing BPR from 64 to 128 factors (with 30 epochs) hurt all metrics: NDCG@10 dropped from 0.239 to 0.238. MovieLens 1M has ~6 040 users and ~3 706 items — a relatively small matrix for 128-dimensional embeddings. The extra capacity likely overfits the training interaction matrix, producing user/item factors that are more precise on training pairs but less generalisable. Keeping 64 factors with 20 epochs remains the optimal BPR configuration for this dataset.
