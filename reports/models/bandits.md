# Multi-Armed Bandits for Model Selection

Important directories:
1. Implementation: `src/models/bandit/` (`strategy.py`, `simulation.py`, `bandit_model_selector.py`)
2. Launch script: `src/run_bandit.py` for full CLI simulation
3. Experiment setup&results: `runs/bandit_*` reports and this document (`reports/models/bandits.md`)

## Motivation

BPR and ItemGraph score nearly identical NDCG@10 (0.296 vs 0.288) on the held-out validation set. However, NDCG is an offline proxy — the true per-user reward distributions (e.g. clickthrough rate) may differ between the two models in ways that NDCG cannot capture. A static model selection based on aggregate NDCG would serve BPR to everyone, potentially ignoring users for whom ItemGraph generates better engagement.

Bandits let us discover per-user model preferences adaptively without requiring explicit cohort definitions upfront. Since we do not have access to an online environment, we simulate this process using per-user NDCG@10 as a proxy for clickthrough rate.

## Exploration-Exploitation Trade-Off

Every user routed to the currently-weaker arm is an exploration cost — that user gets a potentially worse recommendation. However, the arm we consider weaker might actually be getting better results, and we just don't have enough data to see it. Exploration helps us get this data.

A standard **A/B test** is itself a bandit algorithm — specifically, _explore-then-commit_: show each option a fixed percentage of the time during a pure exploration period, then deploy the winner for pure exploitation. It does not adapt during the exploration phase.

**Epsilon-greedy** interleaves exploration and exploitation continuously: with probability epsilon it explores (random arm), otherwise it exploits (best arm so far). However, it explores at a _constant rate_ regardless of how confident it is in each arm's estimate. In our simulation (epsilon=0.1), it converged quickly to BPR, with only 4.5% of users seeing ItemGraph.

**Thompson Sampling** adjusts exploration based on two factors: how good each arm's estimated reward is, and how _confident_ we are in that estimate. It samples from each arm's posterior distribution and picks the highest sample. When uncertain about an arm, the posterior is wide and occasionally produces high samples, driving exploration. As evidence accumulates, the posterior narrows and the better arm wins consistently. It converges slower than epsilon-greedy, routing 18.9% of users to ItemGraph before settling on BPR as the stronger arm.

## Results Summary (NDCG@10 as CTR proxy)

| | Epsilon-Greedy | Thompson Sampling |
|---|---|---|
| BPR selections | 95.5% | 81.1% |
| ItemGraph selections | 4.5% | 18.9% |
| ItemGraph mean reward | 0.255 | 0.282 |
| Converged policy reward | 0.297 | 0.296 |

Both strategies converge to similar overall reward, confirming BPR's global dominance under the NDCG proxy. Thompson Sampling converges slower than epsilon-greedy, dedicating more budget to exploration (18.9% vs 4.5% ItemGraph selections) before settling on BPR as the stronger arm.
