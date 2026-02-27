# Two-Tower Transformer Recommender

Important directories:
1. Implementation: `src/models/ANN/two_towers/` (`two_tower_transformer.py`, `user_tower.py`, `item_tower.py`)
2. Launch script: `src/train_two_tower_transformer.py` for training/evaluation
3. Experiment setup&results: `runs/two_tower_transformer_*` reports and this document (`reports/models/DL_recommenders/two_towers.md`)

## Model Choice Justification

The two-tower architecture decouples user and item representations into independent encoders sharing a 128-dimensional L2-normalised scoring space, enabling sub-millisecond inference (0.49 ms/user) via FAISS, with item embeddings pre-computed once and indexed offline.

The user tower is a 2-layer transformer encoder over interaction history (up to 64 items), followed by cross-attention where a learned demographics embedding queries the contextualised history. This lets the model attend selectively to relevant past interactions rather than treating history as a bag of items. The item tower is a lightweight MLP projecting 1536-dim SentenceBERT embeddings end-to-end, avoiding the information loss of a fixed PCA bottleneck.

## Representational Differences vs MF/BPR

MF and BPR learn a single fixed latent vector per user ID — static, unable to capture temporal preference shifts, and undefined for new users with no training interactions.

The two-tower model computes user embeddings dynamically from the interaction sequence. Two users who watched the same films produce similar embeddings regardless of whether either appeared in training. Rating embeddings (1–5 mapped to learned 16-dim vectors) let the transformer distinguish "loved" from "hated" items in history, a signal MF discards. Content features (SentenceBERT) provide item semantics unavailable to pure collaborative filtering.

The trade-off: MF/BPR memorise individual preferences efficiently; the two-tower must _generalise_ from features, requiring more data and compute to match ID-based memorisation capacity.

## Optimisation and Compute Trade-Offs

Training takes ~2 hours on Apple MPS for 30 epochs, versus seconds for BPR's SGD. InfoNCE with in-batch negatives (B=512, \tau=0.1) coupled with AdamW and cosine LR scheduling is quite computationally-expensive. Self-attention is O(n^2) in history length, justifying the cap at 64 items.

At inference both architectures reduce to dot-product lookup — BPR via latent factors, the two-tower via FAISS. Runtime cost is comparable once the index is built.

## Why Performance Improves — and Where It Degrades

| K | NDCG | Precision | Recall | MRR |
|---|------|-----------|--------|-----|
| 10 | 0.137 | 0.100 | 0.063 | 0.217 |
| 20 | 0.151 | 0.094 | 0.077 | 0.227 |

Best NDCG@10 = 0.137 at epoch 15 on the per-user temporal split. This exceeds the content-based baseline (0.128 on the same split), which notably uses a GBR reranker on top of its base retrieval — the two-tower result is from the base retrieval model alone, without any reranking stage. Direct comparison to BPR (0.239) is invalid: BPR uses a global temporal split that excludes ~75% of users and avoids per-user future-leakage constraints.

Training loss continues to decrease after epoch 15 while NDCG degrades (0.137 → 0.132 by epoch 25), indicating overfitting. With only 6,040 users and 439K training samples, model capacity (transformer + cross-attention + MLP) exceeds the available signal. Nearly 49% of users receive zero relevant items in the top-10, suggesting the single-vector-per-user bottleneck fundamentally limits recall for users with diverse tastes.
