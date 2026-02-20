# Heuristic Recommenders Results

We evaluate heuristic recommenders as top-K rankers using NDCG@K, MRR@K, Precision@K, and Recall@K with relevance threshold `rating >= 4.0`.
NDCG and MRR emphasize ranking quality near the top of the list; Precision/Recall capture hit-rate behavior.

## Experimental Setup

- Candidate models:
- `count` (global popularity)
- `recency` (time-decayed popularity)
- `mean_rating` (average rating with min-count filtering)
- `bayesian` (shrinkage toward prior)
- `item_graph`, `pagerank`, `ppr` (graph-based propagation/random-walk signals)
- Validation and test were evaluated in offline ranking mode (`k in {10, 20}`), with `mode="all"`; additional analysis compared `all` vs `warm_only` (at least one train rating >= threshold (default 4.0)) for best-per-family models. Common behavior in both modes: users with no relevant test items (rating >= threshold) are skipped from metric averaging.

## Conceptual Expectations (Before Looking at Numbers)

- Popularity-based methods should be strong under sparse user history because they estimate robust global priors.
- Recency should help when preference drift is present, but can underperform if drift is weak or noisy.
- Mean-rating heuristics are vulnerable to small-sample bias; without strong shrinkage they often rank niche high-variance items too optimistically.
- Graph methods should improve over pure popularity when item-item co-preference structure is informative and sufficiently dense.
- Bayesian shrinkage should dominate plain means, especially in long-tail regions with few ratings.

## Empirical Highlights

### Validation (K=10): strongest overall by NDCG

| Model | NDCG@10 | MRR@10 | Precision@10 | Recall@10 |
|---|---:|---:|---:|---:|
| `count` | **0.2962** | **0.4737** | 0.2621 | 0.0546 |
| `recency (half_life=60)` | 0.2959 | 0.4721 | 0.2595 | 0.0539 |
| best graph (`item_graph`) | 0.2916 | 0.4476 | **0.2663** | **0.0560** |

Interpretation: popularity priors dominated validation ranking quality (NDCG/MRR), while graph diffusion gave slightly better hit-rate style metrics (precision/recall).

### Test (K=10): best-per-model comparison

| Model | NDCG@10 | MRR@10 | Precision@10 | Recall@10 |
|---|---:|---:|---:|---:|
| `item_graph` | **0.2351** | **0.3683** | **0.2127** | **0.0488** |
| `ppr` | 0.2265 | 0.3567 | 0.2035 | 0.0455 |
| `pagerank` | 0.2220 | 0.3505 | 0.1994 | 0.0443 |
| `count` | 0.2173 | 0.3510 | 0.1927 | 0.0421 |
| `recency` | 0.2141 | 0.3472 | 0.1893 | 0.0417 |

Interpretation: graph methods generalized better on test, overtaking popularity baselines.

## Discussion

### 1) Validation-to-test shift is substantial

All model families drop from validation to test, but the drop is larger for popularity-like models than for graph models.
This is consistent with a mild distribution shift where static global priors are less transferable than relational signals.

### 2) Popularity remains a strong baseline, but not a sufficient one

`count` is still very competitive and should remain in any benchmark suite, but test results show that relying only on popularity likely leaves ranking gains on the table.

### 3) Graph signal captures useful structure beyond exposure bias

`item_graph` / `ppr` / `pagerank` improve top-K quality on test. This suggests user-relevant adjacency (co-like/co-consumption) contains predictive information not recoverable from global frequency alone.

### 4) Mean-rating and Bayesian heuristics are weak relative to count/graph

These methods underperform materially in both validation and test, so most likely in this setup, simple shrinkage over scalar item means is not enough to compete with either robust popularity priors or graph propagation.

### 5) K trade-off is consistent with ranking theory

For leading models, moving from K=10 to K=20 increases recall and reduces NDCG.
This is expected: deeper lists retrieve more relevant items, but with weaker ordering quality near the top.

### 6) `skip_rate` is high (around 0.78 to 0.81)

Fraction of predicted users that were skipped most commonly because they have no relevant items in the evaluation split demonstrates that effective evaluation is performed on a small ~20% subset of users with valid ground-truth and eligible candidates.

## Conclusion

The heuristic study supports a clear decision: keep `count` as a robust baseline, but use graph-based heuristics as the strongest non-parametric candidate for downstream comparisons.
