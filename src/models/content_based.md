# Content-Based Filtering: Results and Analysis

## 1. Item Representation

Each movie is represented as a **400-dimensional dense embedding** produced via the
following pipeline:

1. **Field-level encoding** with SentenceBERT (`all-MiniLM-L6-v2`).
   Four textual fields are encoded independently:
   - **Title** -- the movie name, e.g. "Toy Story".
   - **Genres** -- each genre is encoded as "Movie belongs to {genre} genre";
     the per-genre vectors are averaged.
   - **Year** -- encoded as "Released in {year}".
   - **Description** -- the TMDB synopsis.

   Each field produces a 384-dim vector.

2. **Concatenated fusion**: the four 384-dim field vectors are concatenated into a
   1536-dim vector and L2-normalised.

3. **PCA reduction**: the 1536-dim concatenation is reduced to **400 dimensions** via PCA
   (fitted on the full movie catalogue). This preserves the vast majority of variance
   while making retrieval and downstream computation faster.

### Candidate retrieval with FAISS

Given a user profile and a catalogue of ~3,900 movie embeddings, we need to find the most
similar movies efficiently. We use **FAISS** (`IndexFlatIP`) -- Facebook's library for
dense vector similarity search. With L2-normalised vectors, inner product equals cosine
similarity, so FAISS returns approximate nearest neighbours ranked by cosine similarity.

### Popularity baseline

As a non-personalised baseline, we include a **global popularity** scorer that recommends
movies with the highest average rating (filtered to movies with at least 100 ratings to
avoid niche outliers). This baseline uses no user-specific information and serves as the
floor that any personalised model should beat.

### User profile construction

User profiles are constructed as the **weighted mean of liked-movie embeddings**
(ratings > 4.0), with two weighting schemes applied multiplicatively:

- **Non-linear rating weight**: `(max(0.3, rating - 4.0))^2`, so 5-star movies
  contribute ~3.3x more than 4.1-star movies.
- **Recency weight** (optional): `exp(-decay * age_frac)`, where `age_frac` is the
  chronological position of the rating (1 = oldest, 0 = newest). With `decay = 1.3`,
  the oldest ratings contribute ~3.7x less than the newest.

## 2. Similarity Function Comparison

Two similarity functions are compared:

| Similarity | Interpretation |
|---|---|
| **Cosine** | Measures directional alignment of raw embedding vectors. Treats all embedding dimensions equally. |
| **Pearson** | Equivalent to cosine similarity on mean-centred vectors. Removes per-dimension bias, focusing on relative variation patterns. |

### Head-to-head: Cosine vs Pearson

| Configuration | Metric | NDCG@10 | Precision@10 | Recall@10 |
|---|---|---|---|---|
| Similarity only (no enhancements) | Cosine | 0.04845 | 0.03566 | 0.01867 |
| Similarity only (no enhancements) | Pearson | 0.04849 | 0.03582 | 0.01878 |
| + Recency decay (1.3) | Cosine | 0.05248 | 0.03853 | 0.02022 |
| + Recency decay (1.3) | Pearson | 0.05270 | 0.03882 | 0.02040 |
| Hybrid (beta=0.9, no enhancements) | Cosine | 0.06393 | 0.05337 | 0.02636 |
| Hybrid (beta=0.9, no enhancements) | Pearson | 0.06395 | 0.05326 | 0.02630 |

**Observation**: Pearson and Cosine perform almost identically in this setting, with
Pearson holding an edge (~0.1% NDCG). This is expected because PCA already
centres the data to some extent. We use **Pearson** as the default going forward because
the mean-centring is theoretically justified: it makes the similarity measure invariant
to a constant offset in the embedding space, focusing purely on relative feature patterns.

## 3. Experimental Results

All experiments use `k=10`, relevance threshold `4.0`, PCA dim `400`, Pearson metric,
and a per-user temporal 75/25 train/val split.

### Why per-user temporal split

We use a **per-user temporal split** rather than a global temporal split. In the global
split, a single timestamp cutoff divides all ratings into train/val -- but this means
infrequent users may have *all* their ratings land in train (nothing to evaluate) or *all*
in val (no history to build a profile from). The per-user split sorts each user's ratings
chronologically and takes their first 75% as train and last 25% as val. This guarantees
every user has both training history and held-out ratings, making evaluation more reliable
and representative across the full user population.

### Full results table

| # | Configuration | NDCG@10 | P@10 | R@10 |
|---|---|---|---|---|
| 1 | Global popularity baseline | 0.04507 | 0.04448 | 0.01771 |
| 2 | Mean rating (personalised retrieval, global rating score) | 0.04827 | 0.04845 | 0.02162 |
| 3 | Similarity scoring (Pearson) | 0.04849 | 0.03582 | 0.01878 |
| 4 | + Recency decay (1.3) | 0.05270 | 0.03882 | 0.02040 |
| 5 | + Neighbor enrichment (K=12) | 0.06532 | 0.04761 | 0.02501 |
| 6 | Hybrid scoring (beta=0.9, Pearson) | 0.06395 | 0.05326 | 0.02630 |
| 7 | + Recency decay (1.3) | 0.06654 | 0.05599 | 0.02791 |
| 8 | + Neighbor enrichment (K=12) | 0.08578 | 0.06971 | 0.03546 |
| 9 | **GBR re-ranker** (full pipeline) | **0.12822** | **0.10966** | **0.05185** |

### Scoring strategies

- **Popularity baseline**: recommends globally highest-rated movies. No personalisation.
- **Mean rating**: FAISS retrieves personalised candidates, but scores them by global mean rating. The retrieval provides some personalisation, but the scoring ignores similarity.
- **Similarity**: scores candidates by cosine/Pearson similarity to the user profile. Pure content-based signal.
- **Hybrid**: blends similarity and mean rating as `beta * sim + (1-beta) * mean_rating`. With `beta=0.9`, this gives 90% weight to similarity and 10% to a popularity prior, acting as a tiebreaker among similarly-scored items.

### Progressive improvements

- **Recency decay**: weighting recent ratings more heavily when building user profiles. Improves NDCG by ~8% relative over the non-recency baseline, reflecting that user tastes evolve over time.
- **Neighbor enrichment**: averaging each user's profile with their K=12 nearest-neighbor profiles injects a collaborative-filtering signal. This is the single largest improvement for the similarity and hybrid scorers (+28-35% relative NDCG). Intuitively, it addresses the sparsity of individual user profiles by leveraging similar users' tastes.
- **GBR re-ranker**: a `GradientBoostingRegressor` trained pointwise on 6 features (cosine similarity, movie mean rating, movie rating count, user mean rating, user rating count, user liked count). This is the most impactful single addition, lifting NDCG from 0.086 to 0.128 (+49% relative). The tree model learns non-linear interactions between content similarity and popularity signals that the linear hybrid cannot capture.

## 4. Interpretation

The content-based model starts from a modest NDCG of 0.048 with pure similarity scoring.
Each enhancement addresses a specific weakness:

1. **Recency weighting** addresses temporal drift in user preferences. Its effect is
   small but consistent across all scoring strategies.

2. **Neighbor-based profile enrichment** is the most impactful "free" improvement. By
   averaging a user's embedding with their nearest neighbors', we inject collaborative
   signal into a purely content-based profile. This bridges the gap between content-based
   and collaborative filtering without requiring an explicit interaction matrix.

3. **The GBR re-ranker** combines content and popularity signals through a learned
   non-linear model. Feature importance analysis shows that `cosine_similarity` and
   `movie_mean_rating` are the two dominant features, confirming that the model primarily
   learns a more sophisticated version of the hybrid scoring -- but with the flexibility
   to learn non-linear decision boundaries (e.g. "trust similarity more for niche movies,
   trust popularity more for mainstream ones").

The final model (NDCG@10 = 0.128) represents a **2.8x improvement** over the popularity
baseline and a **2.6x improvement** over the pure similarity baseline.

## 5. Commands to Reproduce

```bash
# Popularity baseline
python -m src.run_content_based --scoring popular --metric cosine --recency-decay 0 --n-neighbors 0

# Similarity (Cosine)
python -m src.run_content_based --scoring similarity --metric cosine --recency-decay 0 --n-neighbors 0

# Similarity (Pearson)
python -m src.run_content_based --scoring similarity --metric pearson --recency-decay 0 --n-neighbors 0

# Full pipeline (best config)
python -m src.run_content_based --scoring gbr_reranker --metric pearson --recency-decay 1.3 --n-neighbors 12
```
