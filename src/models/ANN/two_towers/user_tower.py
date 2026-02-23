"""User tower for the two-tower transformer recommender.

Encodes a user's interaction history and demographic features into a single
dense vector in the shared scoring space.  The pipeline is:

1.  Concatenate each history item's SentenceBERT embedding with a learned
    rating embedding, then project into the transformer hidden dimension.
2.  Add sinusoidal positional encoding (position 0 = most recent item).
3.  Self-attend over the history sequence with a lightweight TransformerEncoder.
4.  Cross-attend from a demographics query to the contextualised history.
5.  L2-normalise the output for dot-product scoring against item embeddings.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as functional


# ── Demographic feature encoding constants ───────────────────────────────────

# MovieLens 1M age buckets mapped to contiguous indices.
AGE_BUCKET_TO_INDEX: Dict[int, int] = {
    1: 0,
    18: 1,
    25: 2,
    35: 3,
    45: 4,
    50: 5,
    56: 6,
}

GENDER_TO_INDEX: Dict[str, int] = {"M": 0, "F": 1}

NUMBER_OF_GENDERS = 2
NUMBER_OF_AGE_BUCKETS = 7
NUMBER_OF_OCCUPATIONS = 21
NUMBER_OF_RATING_LEVELS = 5  # ratings 1–5


class DemographicsEncoder(nn.Module):
    """Encode categorical user demographics into a dense query vector.

    Features used (MovieLens 1M):
    - Gender   (2 categories)   → Embedding(2, demographics_embedding_dimension)
    - Age      (7 buckets)      → Embedding(7, demographics_embedding_dimension)
    - Occupation (21 categories) → Embedding(21, demographics_embedding_dimension)

    The three embeddings are concatenated and projected into the transformer
    hidden dimension.

    Formula
    -------
    d_user = GELU(LayerNorm(W_demo * [Emb_g(g); Emb_a(a); Emb_o(o)] + b_demo))
    """

    def __init__(
        self,
        demographics_embedding_dimension: int = 16,
        output_dimension: int = 128,
    ) -> None:
        super().__init__()

        self.gender_embedding = nn.Embedding(
            NUMBER_OF_GENDERS,
            demographics_embedding_dimension,
        )
        self.age_embedding = nn.Embedding(
            NUMBER_OF_AGE_BUCKETS,
            demographics_embedding_dimension,
        )
        self.occupation_embedding = nn.Embedding(
            NUMBER_OF_OCCUPATIONS,
            demographics_embedding_dimension,
        )

        concatenated_dimension = 3 * demographics_embedding_dimension
        self.projection = nn.Sequential(
            nn.Linear(concatenated_dimension, output_dimension),
            nn.LayerNorm(output_dimension),
            nn.GELU(),
        )

    def forward(
        self,
        gender_indices: torch.LongTensor,
        age_indices: torch.LongTensor,
        occupation_indices: torch.LongTensor,
    ) -> torch.Tensor:
        """Encode demographics into a dense vector.

        Parameters
        ----------
        gender_indices : torch.LongTensor
            Shape ``(batch,)``.  0 = Male, 1 = Female.
        age_indices : torch.LongTensor
            Shape ``(batch,)``.  Indices 0–6 corresponding to MovieLens 1M
            age buckets (see ``AGE_BUCKET_TO_INDEX``).
        occupation_indices : torch.LongTensor
            Shape ``(batch,)``.  Values 0–20.

        Returns
        -------
        torch.Tensor
            Demographics embedding, shape ``(batch, output_dimension)``.
        """
        gender = self.gender_embedding(gender_indices)
        age = self.age_embedding(age_indices)
        occupation = self.occupation_embedding(occupation_indices)
        concatenated = torch.cat([gender, age, occupation], dim=-1)
        return self.projection(concatenated)


class SinusoidalPositionalEncoding(nn.Module):
    """Deterministic sinusoidal positional encoding (Vaswani et al., 2017).

    Position 0 corresponds to the **most recent** interaction so that lower
    position indices encode higher recency.  The encoding is pre-computed and
    stored as a non-learnable buffer.

    Formula
    -------
    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """

    def __init__(
        self,
        maximum_history_length: int = 64,
        embedding_dimension: int = 128,
    ) -> None:
        super().__init__()

        positional_encoding = torch.zeros(
            maximum_history_length,
            embedding_dimension,
        )

        position = torch.arange(0, maximum_history_length).unsqueeze(1).float()

        # div_term = 10000^(2i / d_model)
        div_term = torch.exp(
            torch.arange(0, embedding_dimension, 2).float()
            * (-math.log(10000.0) / embedding_dimension)
        )

        # PE(pos, 2i)   = sin(pos / div_term)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        # PE(pos, 2i+1) = cos(pos / div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer so it moves with the model device but is not
        # treated as a learnable parameter.
        self.register_buffer(
            "positional_encoding",
            positional_encoding.unsqueeze(0),  # (1, max_len, dim)
        )

    def forward(self, sequence_length: int) -> torch.Tensor:
        """Return positional encodings for the first ``sequence_length`` positions.

        Parameters
        ----------
        sequence_length : int
            Actual (non-padded) length of the longest sequence in the batch.

        Returns
        -------
        torch.Tensor
            Shape ``(1, sequence_length, embedding_dimension)``, broadcastable
            to ``(batch, sequence_length, embedding_dimension)``.
        """
        return self.positional_encoding[:, :sequence_length, :]


class UserTower(nn.Module):
    """Transformer-based user encoder.

    Produces a single user embedding by:
    1.  Projecting each history item (SentenceBERT ∥ rating embedding) into the
        transformer hidden dimension.
    2.  Adding sinusoidal positional encoding (recency-ordered).
    3.  Running self-attention over the history sequence.
    4.  Cross-attending from a demographics query to the contextualised items.

    Formula
    -------
    x_i = Linear_user([e_i ; Emb_rating(r_i)]) + SinPE(i)
    H   = TransformerEncoder([x_1, ..., x_n])
    z_user = Normalize(LayerNorm(CrossAttention(Q=d_user, K=H, V=H)))

    where e_i is the 1536-dim SentenceBERT embedding, r_i is the rating (1–5),
    Emb_rating maps it to a 16-dim vector, [;] denotes concatenation, and
    SinPE(i) is the sinusoidal positional encoding for recency position i
    (0 = most recent).

    Note: Linear_user projects from 1552-dim (1536 + 16) to 128-dim, which is
    a different learned projection than the item tower's Linear_item
    (1536 → 128).  This allows each tower to learn the most useful
    representation for its purpose.
    """

    def __init__(
        self,
        item_embedding_dimension: int = 1536,
        projection_dimension: int = 128,
        number_of_transformer_layers: int = 2,
        number_of_attention_heads: int = 4,
        feedforward_dimension: int = 256,
        maximum_history_length: int = 64,
        dropout_rate: float = 0.1,
        demographics_embedding_dimension: int = 16,
        rating_embedding_dimension: int = 16,
    ) -> None:
        super().__init__()

        self.projection_dimension = projection_dimension

        # ── Rating embedding ─────────────────────────────────────────────
        # Ratings 1–5 are mapped to indices 0–4.
        self.rating_embedding = nn.Embedding(
            NUMBER_OF_RATING_LEVELS,
            rating_embedding_dimension,
        )

        # ── Input projection ─────────────────────────────────────────────
        # Project [item_embedding ; rating_embedding] → transformer hidden dim.
        input_dimension = item_embedding_dimension + rating_embedding_dimension
        self.input_projection = nn.Linear(input_dimension, projection_dimension)

        # ── Positional encoding (sinusoidal, non-learnable) ──────────────
        self.positional_encoding = SinusoidalPositionalEncoding(
            maximum_history_length=maximum_history_length,
            embedding_dimension=projection_dimension,
        )

        # ── Self-attention over history ──────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=projection_dimension,
            nhead=number_of_attention_heads,
            dim_feedforward=feedforward_dimension,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for more stable training
        )
        self.self_attention_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=number_of_transformer_layers,
        )

        # ── Cross-attention: demographics → history ──────────────────────
        self.demographics_encoder = DemographicsEncoder(
            demographics_embedding_dimension=demographics_embedding_dimension,
            output_dimension=projection_dimension,
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=projection_dimension,
            num_heads=number_of_attention_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        # ── Output normalisation ─────────────────────────────────────────
        self.output_layer_norm = nn.LayerNorm(projection_dimension)

    def forward(
        self,
        item_history_embeddings: torch.Tensor,
        history_rating_indices: torch.LongTensor,
        history_padding_mask: torch.BoolTensor,
        gender_indices: torch.LongTensor,
        age_indices: torch.LongTensor,
        occupation_indices: torch.LongTensor,
    ) -> torch.Tensor:
        """Encode user history and demographics into a single embedding.

        Parameters
        ----------
        item_history_embeddings : torch.Tensor
            Raw SentenceBERT concat embeddings for each history item,
            shape ``(batch, sequence_length, 1536)``.
        history_rating_indices : torch.LongTensor
            Rating indices (0–4, mapping ratings 1–5) for each history item,
            shape ``(batch, sequence_length)``.
        history_padding_mask : torch.BoolTensor
            Padding mask where ``True`` marks padded (invalid) positions,
            shape ``(batch, sequence_length)``.
        gender_indices : torch.LongTensor
            Shape ``(batch,)``.
        age_indices : torch.LongTensor
            Shape ``(batch,)``.
        occupation_indices : torch.LongTensor
            Shape ``(batch,)``.

        Returns
        -------
        torch.Tensor
            L2-normalised user embedding, shape ``(batch, projection_dimension)``.
        """
        batch_size, sequence_length, _ = item_history_embeddings.shape

        # ── 1. Concatenate item embeddings with rating embeddings ────────
        # rating_embedded: (batch, seq, rating_embedding_dim)
        rating_embedded = self.rating_embedding(history_rating_indices)
        # combined: (batch, seq, 1536 + 16 = 1552)
        combined = torch.cat([item_history_embeddings, rating_embedded], dim=-1)

        # ── 2. Project to transformer hidden dimension ───────────────────
        # projected: (batch, seq, projection_dimension)
        projected = self.input_projection(combined)

        # ── 3. Add sinusoidal positional encoding ────────────────────────
        # positional_encoding: (1, seq, projection_dimension)
        positional_encoding = self.positional_encoding(sequence_length)
        projected = projected + positional_encoding

        # ── 4. Self-attention over history ────────────────────────────────
        # history_context: (batch, seq, projection_dimension)
        history_context = self.self_attention_encoder(
            projected,
            src_key_padding_mask=history_padding_mask,
        )

        # ── 5. Cross-attention: demographics query → history ─────────────
        # demographics_query: (batch, projection_dimension)
        demographics_query = self.demographics_encoder(
            gender_indices,
            age_indices,
            occupation_indices,
        )
        # Unsqueeze to (batch, 1, projection_dimension) for attention
        demographics_query = demographics_query.unsqueeze(1)

        # cross_attention_output: (batch, 1, projection_dimension)
        cross_attention_output, _ = self.cross_attention(
            query=demographics_query,
            key=history_context,
            value=history_context,
            key_padding_mask=history_padding_mask,
        )

        # ── 6. Output: squeeze, LayerNorm, L2-normalise ─────────────────
        # user_embedding: (batch, projection_dimension)
        user_embedding = cross_attention_output.squeeze(1)
        user_embedding = self.output_layer_norm(user_embedding)
        return functional.normalize(user_embedding, p=2, dim=-1)
