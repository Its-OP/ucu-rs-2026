# Evaluation Strategy

## 1. Objective

Define a reusable offline evaluation methodology for comparing recommendation models on the MovieLens 1M dataset. This document specifies the data splitting protocol, the evaluation paradigm, the metric suite, and an honest discussion of what this setup can and cannot measure. Every design choice is grounded in the data pathologies identified during EDA (see [`description.md`](./description.md)).

---

## 2. Evaluation Paradigm: Why Ranking

Three paradigms are available for evaluating recommender systems:

| Paradigm | Question it answers | Typical metrics |
|---|---|---|
| **Rating prediction** | "How close is the predicted score to the true score?" | RMSE, MAE |
| **Classification** | "Can we predict whether the user will like/dislike this item?" | Accuracy, F1, AUC |
| **Ranking** | "Does the model place the items the user actually interacted with near the top of the list?" | NDCG@K, Precision@K, Recall@K |

### Our choice: Ranking

We adopt **ranking-based evaluation** as the primary paradigm. The justification follows directly from two EDA findings:

1. **Selection bias (MNAR).** The mean rating is 3.58 — users self-select movies they expect to enjoy. The observed ratings are not a representative sample of the full preference matrix. Rating-prediction metrics (RMSE, MAE) are computed only over these biased observations, rewarding a model that accurately predicts scores on items the user *already chose to watch*. This tells us little about how the model would perform in deployment, where the task is to surface relevant items from the entire catalog, including ones the user has never seen.

2. **Popularity skew (Gini = 0.634).** A model that predicts popular-item ratings well can achieve low RMSE while completely ignoring the long tail. Ranking metrics, especially when computed over a mixed candidate set of positives and unobserved items, penalize a model that cannot distinguish genuine interest from mere popularity.

3. **The real-world task is ranking.** A deployed recommender presents a top-K list; the user never sees a predicted score. Ranking evaluation directly measures the quality of that list.

Classification (like/dislike) was considered but rejected: binarizing a 1–5 scale at a threshold (e.g., >= 4 as "like") discards ordinal information and introduces an arbitrary cutoff. Ranking subsumes this: if all "liked" items are ranked at the top, classification accuracy follows naturally.

---

## 3. Data Split Strategy: Temporal Split

### 3.1 Why Not Random Split

Random splitting violates the temporal structure of the data. The EDA shows that rating volume, new-user arrival, and new-movie emergence all vary across the 35-month collection window. A random split would allow a model to train on a rating from February 2003 and predict one from April 2000 — an information leak that inflates offline metrics and misrepresents deployment performance, where the model must predict future behavior from past data.

### 3.2 Split Mechanics

We use a **global-timestamp temporal split** with three contiguous, non-overlapping windows:

```
Timeline: Apr 2000 ──────────────────────────────────── Feb 2003
          │           TRAIN           │   VAL   │  TEST  │
          │        (~80%)             │ (~10%)  │ (~10%) │
          ◄──────────────────────────►◄────────►◄───────►
                                     t_1       t_2
```

**Procedure:**

1. Sort all 1,000,209 ratings by timestamp.
2. Choose cutoff timestamps `t_1` and `t_2` such that approximately 80% / 10% / 10% of ratings fall into train / validation / test.
3. Assign each rating to a split based solely on its timestamp:
   - **Train**: `timestamp < t_1`
   - **Validation**: `t_1 <= timestamp < t_2`
   - **Test**: `timestamp >= t_2`

**Properties:**

- No temporal leakage: the model never sees future interactions during training.
- The validation set is used for hyperparameter tuning (latent dimension, regularization strength, number of neighbors, K). The test set is held out until final reporting.
- Some users/items in val/test may have zero training interactions (cold-start). These are retained — not filtered out — because cold-start is a real deployment scenario (12.4% of users, 12.0% of items per EDA).

### 3.3 Relevance Definitions

Our metric suite uses two relevance schemes — **graded** and **binary** — depending on the metric:

**Graded relevance (for NDCG@K).** Each test/val rating retains its original 1–5 score as the relevance label. This lets NDCG distinguish a 5-star item ranked first from a 4-star item ranked first: both are good, but the former is better. The ideal ranking sorts by rating descending, fully preserving ordinal preference information.

**Binary relevance (for Precision@K and Recall@K).** These metrics require a relevant/non-relevant label. We apply a threshold:

- A test/val rating is **relevant** if `rating >= 4` (the user genuinely liked the item).
- Ratings of 1–3 in the test set are treated as **non-relevant** observed interactions.

This threshold is motivated by the EDA's rating distribution: the mean is 3.58 and the distribution is skewed toward 4–5. A threshold of 4 separates genuine positive signal from neutral/lukewarm ratings. A user who rated a movie 3 in a dataset where the average is 3.58 is expressing below-average enthusiasm.

Using graded relevance for the primary metric and binary for the secondary metrics gives us the best of both worlds: NDCG captures fine-grained ranking quality, while Precision and Recall provide crisp, interpretable diagnostics about whether "good" items make it into the top-K.

### 3.4 Candidate Generation for Per-User Ranking

For each user in the val/test set, we rank a candidate set consisting of:

- All items the user has **not** interacted with in the training set (unobserved items).
- Plus the user's actual val/test items (both relevant and non-relevant).

The model produces a score for every candidate, we sort by score descending, and compute metrics on the top-K positions.

---

## 4. Metrics

### 4.1 Primary Metric: NDCG@K (Normalized Discounted Cumulative Gain)

**Definition.** For a given user, let `rel_i` be the **graded relevance** (the original 1–5 rating) of the item at position `i` in the model's ranked list:

```
DCG@K  = Σ (i=1..K)  rel_i / log2(i + 1)
IDCG@K = DCG@K for the ideal ranking (items sorted by rating descending)
NDCG@K = DCG@K / IDCG@K
```

The final NDCG@K is averaged across all users in the evaluation set.

**Why graded relevance.** Unlike binary relevance, graded NDCG distinguishes between ranking a 5-star item at position 1 versus a 3-star item at position 1. The ideal ranking places the user's highest-rated items first, then progressively lower-rated ones. This preserves the full ordinal information in the 1–5 scale rather than collapsing it into a like/dislike binary. In a dataset where the selection bias compresses most ratings into the 3–5 range (EDA: mean = 3.58), this granularity matters — a model that consistently surfaces 5-star items over 4-star items is meaningfully better than one that treats them identically.

**Why primary.** NDCG is position-sensitive: placing a high-relevance item at rank 1 contributes far more than placing it at rank 10. This directly models user behavior — users inspect the top of the list first and attention decays rapidly. Among ranking metrics, NDCG is the most informative single number because it rewards both the *presence* and the *ordering* of relevant items.

**Choice of K.** We report K = 10 as the default (a typical "top page" of recommendations). Sensitivity analysis at K ∈ {5, 10, 20} will be reported to check stability.

### 4.2 Secondary Metric: Precision@K

**Definition.** Using **binary relevance** (rating >= 4), the fraction of the top-K recommended items that are relevant:

```
Precision@K = |{relevant items in top-K}| / K
```

**Role.** Precision answers: "Of what I showed the user, how much was useful?" It is a stricter measure than NDCG because it does not give partial credit for relevant items at lower positions — either an item is in the top-K or it is not. This makes it a useful diagnostic for checking whether the model's top-K list is "clean" (low noise).

**Limitation.** Precision ignores how many relevant items exist. A user with 50 relevant items and another with 2 are judged on the same K-sized window, which can disadvantage models that spread relevant items broadly. This is why Precision is secondary, not primary.

### 4.3 Secondary Metric: Recall@K

**Definition.** Using **binary relevance** (rating >= 4), the fraction of a user's relevant items that appear in the top-K list:

```
Recall@K = |{relevant items in top-K}| / |{all relevant items for this user}|
```

**Role.** Recall answers: "Of everything the user likes, how much did I surface?" It complements Precision by measuring coverage of the user's preference set. In our dataset, where user activity varies wildly (20 to 2,314 ratings), Recall@K reveals whether the model captures the breadth of a user's taste or only their most obvious preferences.

**Limitation.** Recall@K penalizes users with many relevant items (harder to capture them all in K slots) and can be trivially 1.0 for users with very few relevant items. It also does not account for the ordering within the top-K. This is precisely why it pairs with NDCG (order-aware) and Precision (purity-aware).

### 4.4 Metric Summary

| Metric | Type | Relevance | What it measures | Sensitive to position? |
|---|---|---|---|---|
| **NDCG@K** | Primary | Graded (1–5) | Ranking quality with position discount | Yes |
| **Precision@K** | Secondary | Binary (>= 4) | Purity of the top-K list | No (set-based) |
| **Recall@K** | Secondary | Binary (>= 4) | Coverage of user's relevant items | No (set-based) |

All three metrics are reported as averages across users. To account for the power-law user activity distribution (EDA Section 3), we also report metrics **per user-activity segment** (low / medium / high / power) so that strong performance on power users does not mask degradation on the majority.

---

## 5. Diagnostic Breakdowns

Beyond aggregate metrics, we mandate the following sliced evaluations (motivated directly by EDA pathologies):

| Slice | Motivation (EDA finding) | What it reveals |
|---|---|---|
| **User activity tier** (low / medium / high / power) | 6.6% power users contribute 29% of ratings | Whether the model only works for data-rich users |
| **Item popularity tier** (head / torso / tail) | Top 20% of movies hold 65% of ratings (Gini = 0.634) | Whether the model degenerates into a popularity ranker |
| **Cold-start vs. warm-start** (users and items with < threshold training interactions) | 12% of entities are cold | Whether the hybrid/content-based component actually helps |

These breakdowns are not separate metrics — they are the same NDCG@K / Precision@K / Recall@K computed on subsets of the test set. They provide the diagnostic granularity needed to identify where a model succeeds and where it fails.

---

## 6. What This Evaluation Setup Captures

1. **Temporal realism.** By splitting on time, we simulate real deployment: the model is trained on the past and tested on the future. Metrics reflect how well the model would have performed had it been deployed at timestamp `t_1`.

2. **Ranking quality.** NDCG@K directly measures whether relevant items appear at the top — the actual task of a recommender. Precision and Recall provide complementary views of list purity and preference coverage.

3. **Pathology-aware diagnostics.** Slicing by user activity, item popularity, and cold-start status ensures we detect failure modes that aggregate metrics would conceal (a model scoring NDCG = 0.40 overall might score 0.55 on power users and 0.15 on cold-start users).

4. **Hyperparameter selection.** The three-way split (train/val/test) allows tuning on validation without contaminating test results. The validation set also respects temporal ordering.

5. **Model-agnostic.** The protocol applies identically to item-kNN, user-kNN, matrix factorization (SVD, ALS, BPR), factorization machines, and hybrid content-based models. All produce a per-user score for each candidate item, which is ranked and evaluated.

## 7. What This Evaluation Setup Fails to Capture

1. **True negative preferences.** We only observe ratings the user chose to give (MNAR). The test set is biased toward items the user selected — we cannot know how the user would rate a random item they never encountered. This means all our metrics are computed on a non-representative slice of the full preference space. Unbiased evaluation would require randomized exposure (e.g., serving random recommendations and collecting feedback), which is impossible offline.

2. **Beyond-accuracy qualities.** Our metrics measure relevance of individual items but not:
   - **Diversity**: does the top-K list cover multiple genres/moods, or is it monotonous?
   - **Novelty**: does the model surface surprising items, or only well-known ones?
   - **Serendipity**: does the model introduce the user to items outside their comfort zone?
   - **Catalog coverage**: what fraction of the 3,883 movies ever gets recommended to anyone?
   These are critical for user satisfaction in practice but orthogonal to the ranking metrics we measure.

3. **User satisfaction and engagement.** Offline ranking metrics are a proxy. A user might prefer a list with slightly lower NDCG that feels more varied over a technically optimal but repetitive list. Online A/B testing is the only ground truth for user satisfaction.

4. **Feedback loop effects.** In deployment, recommendations influence what users watch, which influences future training data. Our offline evaluation assumes the test set is generated independently of the model, which is true here (MovieLens collected data before any of our models existed) but would not hold in a live system.

5. **Position bias in ground truth.** MovieLens ratings are explicit (users actively chose to rate), so position bias (users only clicking top results) is less of a concern here than in implicit-feedback datasets. However, the selection bias discussed in the EDA still applies: users only rate what they chose to watch.

6. **Cold-start ceiling.** We retain cold-start users/items in the test set (correctly), but a purely collaborative model will necessarily score near zero for them. This is not a flaw in evaluation — it accurately reflects the model's limitation — but it means aggregate metrics will always be dragged down by cold-start entities, even for an otherwise excellent model.

---

## 8. Summary

| Aspect | Decision | Justification |
|---|---|---|
| **Paradigm** | Ranking | Matches deployment task; robust to selection bias (MNAR) |
| **Split** | Temporal (80/10/10 by timestamp) | Prevents temporal leakage; reflects real-world chronology |
| **Relevance** | Graded (1–5) for NDCG; binary (>= 4) for Precision/Recall | NDCG preserves ordinal information; binary threshold separates genuine positive signal from lukewarm (mean 3.58) |
| **Primary metric** | NDCG@10 (graded) | Position-aware ranking quality with full ordinal sensitivity |
| **Secondary metrics** | Precision@10, Recall@10 | List purity and preference coverage |
| **Diagnostic slices** | User activity, item popularity, cold-start | Exposes pathology-specific failure modes |
| **Candidate set** | All unobserved + test items per user | Full-catalog ranking, not sampled negatives |
