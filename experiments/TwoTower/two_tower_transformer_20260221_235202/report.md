# Two-Tower Transformer — Training Report

Generated: 2026-02-22T00:11:08.328978

## Configuration

| Parameter | Value |
|-----------|-------|
| batch_size | 64 |
| dataset_size | 439190 |
| device | cpu |
| dropout_rate | 0.1 |
| evaluation_interval | 1 |
| evaluation_ks | [10, 20] |
| feedforward_dimension | 256 |
| item_hidden_dimension | 256 |
| learning_rate | 0.001 |
| maximum_history_length | 64 |
| number_of_attention_heads | 4 |
| number_of_epochs | 2 |
| number_of_movies | 3883 |
| number_of_transformer_layers | 2 |
| number_of_users | 6040 |
| positive_threshold | 4.0 |
| projection_dimension | 128 |
| random_seed | 42 |
| temperature | 0.1 |
| weight_decay | 1e-05 |

## Training History

| Epoch | Loss | LR | NDCG@10 | Precision@10 | Recall@10 |
|-------|------|----|---------|-------------|-----------|
| 1 | 3.141385 | 5.87e-04 | 0.07270 | 0.05287 | 0.03010 |
| 2 | 2.730187 | 0.00e+00 | 0.07781 | 0.05520 | 0.03246 |

## Best Model

- **Epoch:** 2
- **NDCG@10:** 0.07781

## Final Evaluation (Offline)

| K | NDCG | Precision | Recall | MRR | MAP |
|---|------|-----------|--------|-----|-----|
| 10 | 0.07781 | 0.05520 | 0.03246 | 0.13095 | 0.02689 |
| 20 | 0.08843 | 0.05306 | 0.04173 | 0.14109 | 0.02597 |

## Performance

- **Total training time:** 1132.2 seconds
- **Total prediction time:** 3.7 seconds (6040 users)
- **Time per user:** 0.62 ms
