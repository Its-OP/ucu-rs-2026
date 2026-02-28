# Bandit Model Selector — Simulation Report

Generated: 2026-02-23T23:47:59.770918

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
| prior_alpha | 1.0 |
| prior_beta | 1.0 |
| process_order | temporal |
| relevance_threshold | 4.0 |
| reward_threshold | 0.0 |
| seed | 42 |
| split | val |
| strategy | thompson |

## Individual Arm Baselines (standalone, pre-bandit)

| Arm | NDCG@10 | Precision@10 | Recall@10 |
|-----|---------|-------------|-----------|
| BPR | 0.29635 | 0.26496 | 0.05578 |
| ItemGraph | 0.28831 | 0.26469 | 0.05592 |

## Per-Arm Bandit Results

| Arm | Selections | Selection % | Mean NDCG@10 |
|-----|-----------|-------------|--------------|
| BPR | 964 | 81.1% | 0.29867 |
| ItemGraph | 224 | 18.9% | 0.28150 |

## Converged Policy Evaluation

| Metric | Value |
|--------|-------|
| NDCG@10 | 0.29631 |
| Precision@10 | 0.26496 |
| Recall@10 | 0.05578 |

## Summary

- **Total users processed:** 1188
- **Users skipped (no relevant items):** 35
- **Overall mean reward (NDCG):** 0.29543

## Final Arm Statistics

| Arm | Pull Count | Reward Sum | Mean Reward |
|-----|-----------|-----------|-------------|
| BPR | 964 | 287.9174 | 0.29867 |
| ItemGraph | 224 | 63.0557 | 0.28150 |
