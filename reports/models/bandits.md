# Multi-Armed Bandits for Model Selection

Important directories:
1. Implementation: `src/models/bandit/` (`strategy.py`, `simulation.py`, `bandit_model_selector.py`)
2. Launch script: `src/run_bandit.py` for full CLI simulation
3. Experiment setup&results: `runs/bandit_*` reports and this document (`reports/models/bandits.md`)

## Why Offline Metrics Are Insufficient

BPR and ItemGraph score nearly identical NDCG@10 (0.296 vs 0.288) on the held-out validation set. A single aggregate number hides _who_ each model serves well. If ItemGraph outperforms BPR for a portion of users but underperforms for the rest, the averages look similar while both models leave value on the table. Offline evaluation tells us which model is better _on average_ — bandits tell us which model is better _per user_.

A static model selection (pick the one with the higher average) would serve BPR to everyone and ignore the subpopulation where ItemGraph wins. Bandits discover these subpopulations adaptively without requiring explicit cohort definitions upfront.

## Exploration-Exploitation Trade-Off

Every user routed to the currently-weaker arm is an exploration cost — that user gets a potentially worse recommendation. But without exploration, we never learn whether the weaker arm is actually better for certain users, and we converge prematurely on a suboptimal policy.

**Epsilon-greedy** (epsilon=0.1) handles this simply: 10% of users are randomly assigned to a uniformly-chosen arm regardless of evidence. This leads to minimal exploration of ItemGraph (4.5% selections) because random assignment wastes exploration budget on users where BPR already dominates.

**Thompson Sampling** explores more efficiently by sampling from each arm's posterior. It naturally explores more when uncertain and exploits when confident, routing 18.9% of users to ItemGraph — nearly 4x more than epsilon-greedy — while maintaining a higher per-arm NDCG for those users (0.282 vs 0.255).

## Results Summary

| | Epsilon-Greedy | Thompson Sampling |
|---|---|---|
| BPR selections | 95.5% | 81.1% |
| ItemGraph selections | 4.5% | 18.9% |
| ItemGraph mean NDCG | 0.255 | 0.282 |
| Converged NDCG@10 | 0.297 | 0.296 |

Both strategies converge to similar overall NDCG, confirming BPR's global dominance. The key difference is, Thompson Sampling delivers higher quality to the users it routes to ItemGraph, suggesting it identifies a genuine subpopulation rather than exploring randomly.
