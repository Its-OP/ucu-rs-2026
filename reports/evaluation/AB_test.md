# A/B Test Plan: BPR vs ItemGraph

## Hypothesis

BPR and ItemGraph produce similar offline NDCG@10 (0.296 vs 0.288). Still, the proxy metric of NDCG might not fully capture the users' behavior, and one model might lead to significantly more clicks than the other. We hypothesize that BPR achieves higher mean CTR than ItemGraph online.

**H0:** mean CTR(BPR) <= mean CTR(ItemGraph)
**H1:** mean CTR(BPR) > mean CTR(ItemGraph)

## Groups

| Group | Model | Role |
|-------|-------|------|
| **Control** | ItemGraph | Baseline |
| **Treatment** | BPR | Challenger |

## Unit of Randomization

**User-level.** Each user is assigned to one group via a hash of UserID.

## Sample Size

Offline evaluation used ~4,000 users. For the online test, assuming baseline CTR ~3%, MDE of 1 percentage point, alpha = 0.05, power = 0.80, we need ~**3,800 users per group** (~7,600 total).

## Statistical Test

One-sided **t-test** on per-user CTR, with significance threshold p < 0.05. At this sample size the t-distribution asymptotically approaches the normal, making it equivalent to a z-test in practice.

## Metrics

**Primary:** CTR (clickthrough rate).

**Guardrail metrics:**
- Session duration — ensures that the users watch the movies they click

## Risks

| Risk | Impact | Mitigation                                                                                           |
|------|--------|------------------------------------------------------------------------------------------------------|
| **Novelty effect** | Users may click more on BPR simply because recommendations feel different, inflating short-term CTR | Run long enough for novelty to wear off; compare early vs late CTR                                   |
| **Position bias** | Users click higher-ranked items regardless of relevance, biasing CTR equally in both groups | Add noise to the ratings before displaying them to the users; adjust number of clicks for the rating |
| **Feedback loops** | Treatment clicks feed into future training, biasing BPR toward its own recommendations | Freeze models during the test; retrain only after experiment concludes                               |
