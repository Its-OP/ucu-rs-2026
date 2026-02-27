# Summary Report — HW#1 Recommender System

## What we actually built
We implemented five model families: a simple popularity baseline, item‑item collaborative filtering
(tested with cosine, adjusted cosine, and Pearson similarity), two matrix‑factorization approaches
(FunkSVD and ALS), and a content‑based pipeline built on SBERT + PCA embeddings with FAISS
retrieval and multiple scoring options, including an optional GBR re‑ranker.

## First, an important caveat (this affects every comparison)
We used **two different temporal splits**:
- **Global temporal split (75/12.5/12.5)** for ALS / FunkSVD / Item‑Item CF.
- **Per‑user temporal split (75/25)** for content‑based.

These are *not* the same problem:
- The **global split** creates cold‑start users/items (some entities only appear in future windows).
- The **per‑user split** guarantees every user has history, which makes the task easier and removes within‑user leakage.

So **we should not compare raw metrics across models** until all models run on the same split and the same candidate set.


## 1) Performance comparison (reasoned, not raw ranking)
Given the split mismatch, the best we can do is compare **expected behavior** on a unified split:

If everything were re‑evaluated on one consistent split, we would expect the following pattern:
matrix factorization (FunkSVD/ALS) usually leads on warm users, item‑item CF is a strong and
interpretable baseline but degrades for sparse users and tail items, content‑based (especially
with the GBR re‑ranker) is the safest option for cold‑start users/items, and the popularity baseline
remains stable but non‑personalized, useful mainly as a fallback.

**Simple takeaway:** MF for warm users, content‑based for cold‑start, CF as a solid baseline.


## 2) Where each model fails

### Popularity baseline
It recommends the same items to everyone, which means it does not personalize at all and
collapses the long tail by definition.

### Item‑Item CF
It breaks for cold‑start users/items, struggles when histories are sparse or noisy, and tends
to amplify popularity skew because similarity is driven by co‑ratings on head items.

### FunkSVD (SGD MF)
It cannot handle cold‑start, and its performance is sensitive to learning rate and regularization,
which can cause it to overfit head items if not tuned carefully.

### ALS (explicit MF)
It also cannot handle cold‑start. Results depend heavily on factor count and regularization,
and it still leans toward popular items under explicit‑only feedback.

### Content‑based (embedding retrieval + re‑ranking)
It is only as good as the embeddings. If they do not capture taste, it fails. It also struggles
when users have too few liked items to build a profile, and it can over‑recommend content‑similar
items that do not match preference.


## 3) Bias analysis (what we should watch for)

### Popularity bias
All collaborative methods naturally prefer head items because those items dominate signal.
If we care about long‑tail exposure, we need explicit mitigation.

### Activity bias
Power users dominate the data, so CF/MF models often perform well for them and poorly for
low‑activity users.

### Cold‑start bias
Global temporal splits produce real cold‑start cases. Collaborative models fail here unless
paired with content‑based or a popularity fallback.


## 4) What would we deploy (and why)

### Recommended: a **hybrid pipeline**
Use MF (ALS or FunkSVD) for warm users/items, fall back to content‑based for cold‑start, and
keep a popularity baseline as a safety net for extreme sparsity.

It gives the best quality where data exists, and still produces reasonable recommendations when it doesn’t.

### If we must pick one model
Given the current results on the same split (global temporal) FunkSVD clearly outperforms ALS,
so FunkSVD is the better MF choice today. If the environment is heavily cold‑start, content‑based
is safer.


## 5) Next steps

### A) Model improvements
Add a re‑ranker (GBR or a lightweight learning‑to‑rank layer), then scale to a bigger
dataset to stress‑test generalization. After that, blend MF scores with content similarity,
add de‑biasing or re‑ranking for long‑tail coverage, and use validation‑based early stopping
for MF tuning.

### B) Add diagnostic slices
Report metrics by user activity tier, by item popularity tier, and by cold‑start vs warm‑start
segments to make failure modes visible.

### C) Product‑level metrics
Track diversity, novelty, and catalog coverage alongside NDCG.
