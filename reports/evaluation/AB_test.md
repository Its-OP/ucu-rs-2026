# A/B Test Plan: Warm vs Cold User Routing

## Hypothesis

Routing cold users (no training interactions with rating >= 4.0) to ItemGraph and warm users to BPR improves overall NDCG compared to BPR-only. BPR dominates globally (NDCG@10 = 0.296 vs 0.288), but Thompson Sampling routed ~19% of users to ItemGraph with higher per-arm NDCG (0.282 vs 0.255 under epsilon-greedy), suggesting ItemGraph better serves cold users who lack the interaction history BPR depends on.

## Groups

| Group | Warm Users | Cold Users |
|-------|-----------|------------|
| **Control** | BPR | BPR |
| **Treatment** | BPR | ItemGraph |

## Unit of Randomization

**User-level.** Each user is assigned to one group via a deterministic hash of UserID + salt, ensuring stable assignment across sessions. A 50/50 split gives ~3,020 users per group.

## Metrics

**Primary:** NDCG@10 (overall, averaged across all users in each group).

**Guardrail metrics:**
- Recall@10 — ensures we don't sacrifice breadth for ranking quality
- Cold-user NDCG@10 — verifies cold users specifically benefit (not just diluted by warm-user dominance)
- Coverage rate — fraction of users receiving a full top-10 list

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Novelty effect** | Cold users may engage more with ItemGraph simply because recommendations feel different, inflating short-term NDCG | Run long enough for novelty to wear off; compare early vs late metrics |
| **Position bias** | Users interact more with higher-ranked items regardless of relevance, biasing NDCG equally in both groups | Same k=10 list length in both groups; report MRR@10 as secondary diagnostic |
| **Feedback loops** | Treatment interactions feed into future training, potentially degrading BPR's cold-user performance over time | Freeze models during the test; retrain only after the experiment concludes |
