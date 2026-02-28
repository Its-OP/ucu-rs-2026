# Summary Report - HW#2 Recommender System

## Implemented components

Seven components were added in the second part, layered on top of the first-part infrastructure:

- **Heuristic rankers** — seven non-parametric models: global count, recency-decayed count,
  mean-rating, Bayesian-shrinkage mean, item-graph diffusion, PageRank, and Personalized PageRank
  (PPR). They require no training and serve as the non-parametric ceiling.

- **BPR** (`BPRRecommender`) — Bayesian Personalized Ranking with matrix-factorization scoring
  (`x_ui = b_i + p_u · q_i`). Trained with SGD on implicit pairwise triplets (user, positive,
  negative). Two negative-sampling strategies (uniform / popularity-weighted) and two negative
  pools (unseen / non-positive) were swept. Best configuration: uniform negatives, unseen pool.

- **Two-stage hybrid** (`TwoStageHybridRecommender`) — BPR retrieves 400 CF candidates; the
  content-based model (SBERT + FAISS) retrieves 120 CB candidates; the union pool (~520 items/user)
  is re-ranked by either RRF or a LambdaMART ranker (`lgb.LGBMRanker`, 16 features, λ-rank
  objective). A post-prediction CF blend (`α · ranker + (1−α) · BPR`) anchors the final score.

- **Two-tower transformer** — decoupled user and item encoders in a shared 128-dim L2-normalised
  space. The user tower is a 2-layer transformer over the last 64 interactions followed by
  cross-attention with a demographics embedding. The item tower is an MLP over 1536-dim SBERT
  embeddings. Trained with InfoNCE / in-batch negatives (B=512, τ=0.1) on Apple MPS.
  Inference via FAISS: 0.49 ms/user after index build.

- **Wide & Deep** (`WideAndDeepRecommender`) — a wide linear branch (global bias, user/item
  biases, demographic biases, genre linear term) combined with a deep MLP branch
  (user/item embeddings + demographic/genre embeddings → 2-layer ReLU MLP). Trained on
  sampled BCE with AdamW; best-epoch selection via held-out validation NDCG.

- **A/B test plan** — a designed online experiment comparing BPR (treatment) vs ItemGraph
  (control) on CTR, with user-level randomisation, one-sided t-test (p < 0.05), ~3 800 users
  per arm, and guardrail metrics (session duration). Not executed online; plan only.

- **Multi-armed bandits** — offline simulation comparing Epsilon-Greedy (ε=0.1) and Thompson
  Sampling for adaptive model selection between BPR and ItemGraph. Per-user NDCG@10 serves as
  the CTR proxy.


## Critical caveats

### Two evaluation protocols are not directly comparable

| Protocol | Train/Val split | Evaluated users | Typical skip rate |
|---|---|---|---|
| **Global temporal** (75/12.5/12.5) | Single cut over all ratings by timestamp | ~1 188–1 347 per split | ~0.78–0.80 |
| **Per-user temporal** (75/25 per user) | Each user's own chronological tail | ~5 999–6 040 | ~0.007 |

The global split evaluates only the ~20 % of users who have ≥4-star ratings inside the narrow
global test window — a high-engagement subset. The per-user split evaluates almost everyone
because every user contributes a personal holdout tail. **Do not compare raw NDCG numbers
across protocols.**

### Validation can overfit to a high-engagement cohort on the global split

The global val and test sets have different user populations. The best global-val config
(RRF no-ranker, NDCG@10=0.296) drops to 0.237 on global test — a 6-point absolute gap.
Any model selected purely on global validation should be treated with caution.

### Bandit and A/B results are simulation-only

Neither the A/B test nor the bandit was run in a real online environment. All reward signals
are offline NDCG@10 proxies; actual CTR behavior may differ.


## 1) Performance comparison

### Global test (same split, held-out, final)

| Model | NDCG@10 | MRR@10 | Recall@10 | MAP@10 |
|---|---:|---:|---:|---:|
| `popularity_count` (baseline) | 0.2173 | 0.3510 | 0.0421 | 0.1233 |
| `item_graph` (α=0.85, steps=1) | 0.2351 | 0.3683 | 0.0488 | 0.1399 |
| `BPR` (uniform, unseen) | 0.2391 | 0.3845 | 0.0504 | 0.1364 |
| **Hybrid LambdaMART blend=0.6** | **0.2476** | **0.3933** | 0.0552 | — |

BPR is ahead of item_graph on NDCG and Recall; item_graph leads on MAP (slightly better average
precision at top-10). The hybrid with LambdaMART and CF blend is the clear winner:
+3.6 % NDCG and +2.3 % MRR over BPR. Pure LambdaMART (blend=1.0) underperforms BPR due to
training-pool covariate shift (200 vs 400 CF candidates at inference) and very low positive
rate (~2 %), which starves LambdaGrad updates of signal.

### Per-user validation (different task geometry, wider user coverage)

| Model | NDCG@10 | MRR@10 | Recall@10 |
|---|---:|---:|---:|
| Default CB baseline | 0.0527 | — | 0.0204 |
| Hybrid CB (FAISS + GBR, part 1) | 0.1282 | — | 0.0519 |
| Wide & Deep `wd_e96` | 0.1276 | 0.2389 | 0.0485 |
| Two-tower (epoch 15) | 0.1373 | 0.2173 | 0.0628 |
| **Hybrid RRF no-ranker (α=0.85)** | **0.1429** | **0.2653** | **0.0575** |

On the per-user split, the LambdaMART variants all fall below the simple RRF no-ranker.
The reranker trains on only ~14 interactions/user after the 10 % holdout carve, too sparse
to learn robust preferences. RRF, relying on BPR's rank and the CB cosine score, benefits
from the richer 90 % training signal and wins comfortably.


## 2) Failure modes by component

### Heuristics (count / recency / graph)
Low complexity, zero training cost, strong non-parametric baseline. The main failure modes:
- **No personalisation** — every user gets the same or popularity-ordered list.
- **Validation-to-test degradation** — popularity priors overfit the high-engagement val subset;
  graph methods generalise better (item_graph NDCG@10 jumps from 3rd on val to 1st on test
  among heuristics). Mean-rating and Bayesian heuristics underperform both count and graph.
- **Tail items are invisible** — item_graph and count score zero on the tail segment.

### BPR
Best pure collaborative model, but:
- **Head-item bias** — head performance (NDCG@10≈0.253) dominates; tail NDCG@10≈0.001, near-zero
  for all methods including BPR (head = top 20 % items by training count, 730 of 3 651 items).
- **Cold-start** — users absent from training fall back to global popularity scores.
- **Negative sampling matters** — uniform negatives outperform popularity-weighted by ~0.014
  NDCG@10 on validation. The sampling strategy effect is larger than unseen vs non-positive pool.

### Two-stage hybrid
Works well when the CF blend anchors the ranker. Failure modes:
- **Pure LambdaMART (blend=1.0) underperforms BPR** on every split because of training-scope
  mismatch and low positive rate.
- **Val winner (RRF) does not transfer to test** — RRF NDCG@10 goes from 0.296 (global val)
  to 0.237 (global test), below BPR. The val population is a high-engagement subset that makes
  simple rank fusion look better than it is.
- **Cold and sparse users get little benefit** — CB profile is noisy with < 5 liked items;
  the reranker's `is_cold_user` flag carries near-zero split importance.

### Two-tower transformer
- **Overfits after epoch 15** — training loss continues to fall (5.62→4.13 over 30 epochs)
  while NDCG peaks at epoch 15 (0.137) and degrades by epoch 25 (0.132). Model capacity
  (transformer + cross-attention + MLP) exceeds the available signal for 6 040 users.
- **Remains below hybrid / BPR on any fair global-test comparison** — the reported 0.137 is
  on per-user val; BPR's 0.239 is on global test. Direct comparison is invalid.
- **High training cost** — ~2 hours on Apple MPS for 30 epochs vs seconds for BPR SGD.
- **49 % of users receive zero relevant items in top-10** — the single user-vector bottleneck
  limits recall for diverse-taste users.

### Wide & Deep
- **Split-sensitive** — best configuration switches between protocols: `wd_e64_h128x64_neg2`
  wins on global val (NDCG@10=0.291) while `wd_e96_h192x96_neg3` wins on per-user val (0.128).
- **Strongest results are validation-only** — no matching global-test win over hybrid or BPR.
- **Higher training cost than BPR** — ~310 s (global) to ~1 172 s (per-user) vs seconds for BPR.
- **BCE loss is a surrogate** — does not directly optimise global top-K ranking; easy/biased
  negatives limit ranking gains.

### Bandits and A/B
- **Reward proxy is offline NDCG** — does not reflect true CTR or engagement; conclusions are
  tentative until real online signals are collected.
- **A/B test is unexecuted** — sample size calculation (3 800 users per arm, baseline CTR ~3 %,
  MDE 1 pp, α=0.05, power=0.80) is correct, but no live experiment has been run.
- **Both bandit strategies converge to BPR** — epsilon-greedy: 95.5 % BPR, 4.5 % ItemGraph
  (converged reward 0.297); Thompson sampling: 81.1 % BPR, 18.9 % ItemGraph (converged
  reward 0.296). The difference between arms is small enough (~0.014 NDCG@10) that Thompson
  Sampling explores ItemGraph substantially before settling.


## 3) Bias and evaluation risks

### Popularity / head bias
Heuristics, BPR, and the hybrid retrieval stage all concentrate signal on head items. Tail
items (80 % of catalog, 2 921 of 3 651 items) show NDCG@10 ≈ 0.000 for every tested model.
Long-tail coverage requires explicit mitigation — content signals, IPS re-weighting, or
diversity-constrained reranking.

### Activity bias
The global split evaluates mostly users who rated actively in the narrow test window.
Power users who dominate the interaction matrix benefit more from all collaborative models;
the evaluation over-represents them.

### Protocol bias
Model rankings can flip between global and per-user setups. RRF wins per-user; LambdaMART
blend wins global. Neither result should be generalised to the other setting without re-evaluation.

### Reranker training density bias
On the per-user split the LambdaMART holdout carve leaves ~14 interactions/user for reranker
labels, too sparse to distinguish personalisation from popularity. The reranker degenerates
toward item-popularity features (`bpr_item_bias`, `movie_mean_rating`, `log_movie_rating_count`),
which together account for ~33 % of LambdaMART split importance alongside user-activity features
(~37 %). CF personalisation features contribute only ~22 %.


## 4) Evaluator evolution: eval.py -> offline_ranking.py

`src/eval/eval.py` (part 1) evaluates one K cutoff and returns three scalar metrics (NDCG,
Precision, Recall). `src/eval/offline_ranking.py` (part 2, now the default) extends it with:

| Change | Old | New |
|---|---|---|
| K support | single K | multiple K in one pass |
| Metrics | NDCG, Precision, Recall | + MRR, MAP |
| Return type | `Metrics` dataclass (3 scalars) | `EvalReport` with per-K metrics + diagnostics |
| User accounting | only skipped-user log | `n_users_total/with_gt/eligible/evaluated`, `skip_rate` |
| Coverage | none | `coverage_rate`, `avg_list_size` |
| Evaluation mode | always all users | `mode="all"` or `"warm_only"` |
| Warm/cold stats | none | `n_warm_users`, `n_cold_users`, `cold_user_rate` |
| Column config | hardcoded | `user_col`, `item_col`, `rating_col` |

The `skip_rate` field is critical for interpreting global-split metrics: a skip_rate of 0.80
means NDCG is averaged over only ~20 % of users, a much narrower (and higher-engagement)
population than the per-user split's 0.7 % skip_rate.


## 5) Deployment recommendation

### Primary production candidate
Deploy `TwoStageHybridRecommender` with LambdaMART and `ranker_cf_blend=0.6` — it is the
strongest model on the held-out global test split (+3.6 % NDCG@10, +2.3 % MRR@10 over BPR).
The content-based retrieval stage adds +16 % relative recall (0.050→0.058), which the CF blend
preserves in the final ranking.

### Safe fallback stack
Keep BPR and item_graph as fallback / challenger arms:
- **BPR** for strong collaborative warm-user quality with fast SGD training.
- **item_graph** as a robust non-parametric backup (MAP@10 > BPR; no training required).
- **popularity count** as the last-resort cold-start fallback.

### Not ready for primary deployment
Two-tower and Wide & Deep are not ready as primary models. Both overfit on this dataset size
and neither beats the hybrid on a comparable held-out test. Their value will increase with
more data or architectural regularisation (e.g., smaller capacity, stronger dropout,
smaller history length for two-tower).


## 6) Possible next steps

### A) Freeze one primary offline protocol before declaring winners
All future model comparisons should be reported on the same global temporal test split.
Per-user validation is useful for debugging but should not be used for final selection because
it changes the user population and task geometry.

### B) Run the designed A/B test online
Test `hybrid_blend=0.6` vs `BPR` (with `item_graph` as an optional third arm). ~3 800 users
per arm, primary metric CTR, guardrail session duration. Freeze both models during the test;
retrain only after it concludes to avoid feedback-loop bias.

### C) Introduce bandits only after A/B confirms non-inferiority
Once A/B validates that the hybrid arm is no worse than BPR, switch to Thompson Sampling for
adaptive allocation. The offline simulation already shows Thompson Sampling explores the
weaker arm more thoroughly (18.9 % vs 4.5 % for epsilon-greedy) before settling on the
better arm, making it a safer online policy.

### D) Add mandatory diagnostic slices to every model report
Every future evaluation should report NDCG@10 separately for: warm vs cold users, head vs
tail items, and low / medium / high / power user-activity tiers. The current results
(tail NDCG ≈ 0 for all models) expose a critical gap that aggregate metrics hide.

### E) Address the long-tail problem explicitly
Add IPS re-weighting or a dedicated tail-aware loss, surface tail items via content-based
or knowledge-graph augmentation, and use diversity-constrained reranking (e.g., MMR or
a coverage-penalised objective) to lift catalog coverage beyond the current head-dominated
recommendations.
