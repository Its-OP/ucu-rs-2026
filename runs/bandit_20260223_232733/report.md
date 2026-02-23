# Bandit Model Selector — Simulation Report

Generated: 2026-02-23T23:33:36.668650

## Configuration

| Parameter | Value |
|-----------|-------|
| bpr_lr | 0.01 |
| bpr_n_epochs | 20 |
| bpr_n_factors | 64 |
| bpr_regularization | 0.01 |
| epsilon | 0.1 |
| graph_alpha | 0.85 |
| graph_steps | 2 |
| graph_threshold | 4.0 |
| k | 10 |
| process_order | temporal |
| relevance_threshold | 4.0 |
| seed | 42 |
| split | val |
| strategy | epsilon_greedy |

## Individual Arm Baselines (standalone, pre-bandit)

| Arm | NDCG@10 | Precision@10 | Recall@10 |
|-----|---------|-------------|-----------|
| BPR | 0.29635 | 0.26496 | 0.05578 |
| ItemGraph | 0.28831 | 0.26469 | 0.05592 |

## Per-Arm Bandit Results

| Arm | Selections | Selection % | Mean NDCG@10 |
|-----|-----------|-------------|--------------|
| BPR | 1135 | 95.5% | 0.29801 |
| ItemGraph | 53 | 4.5% | 0.25450 |

## Converged Policy Evaluation

| Metric | Value |
|--------|-------|
| NDCG@10 | 0.29676 |
| Precision@10 | 0.26532 |
| Recall@10 | 0.05602 |

## Summary

- **Total users processed:** 1188
- **Users skipped (no relevant items):** 35
- **Overall mean reward (NDCG):** 0.29607

## Final Arm Statistics

| Arm | Pull Count | Reward Sum | Mean Reward |
|-----|-----------|-----------|-------------|
| BPR | 1135 | 338.2414 | 0.29801 |
| ItemGraph | 53 | 13.4886 | 0.25450 |
