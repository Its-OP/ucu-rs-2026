"""Item tower for the two-tower transformer recommender.

Projects raw SentenceBERT concatenated embeddings (1536-dim) into a shared
low-dimensional space (default 128-dim) via a learned MLP.  Trained end-to-end
with the user tower so that the projection is optimised for the ranking loss,
replacing the static PCA reduction used elsewhere in the project.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


class ItemTower(nn.Module):
    """Projection MLP that maps raw SentenceBERT embeddings to the shared
    scoring space.

    Architecture
    ------------
    Linear(input_dimension -> hidden_dimension)
        -> LayerNorm -> GELU -> Dropout
        -> Linear(hidden_dimension -> projection_dimension)
        -> LayerNorm -> L2-normalise

    Formula
    -------
    z_item = Normalize(LayerNorm(W2 * GELU(LayerNorm(W1 * e_sbert + b1)) + b2))

    where e_sbert is the 1536-dim concatenated SentenceBERT embedding and
    Normalize denotes L2 normalisation along the feature axis.
    """

    def __init__(
        self,
        input_dimension: int = 1536,
        projection_dimension: int = 128,
        hidden_dimension: int = 256,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dimension = input_dimension
        self.projection_dimension = projection_dimension

        self.projection = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.LayerNorm(hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dimension, projection_dimension),
            nn.LayerNorm(projection_dimension),
        )

    def forward(self, item_embeddings: torch.Tensor) -> torch.Tensor:
        """Project item embeddings into the shared scoring space.

        Parameters
        ----------
        item_embeddings : torch.Tensor
            Raw SentenceBERT concatenated embeddings, shape ``(batch, input_dimension)``.

        Returns
        -------
        torch.Tensor
            L2-normalised projected embeddings, shape ``(batch, projection_dimension)``.
        """
        projected = self.projection(item_embeddings)
        # L2-normalise so that dot product equals cosine similarity
        return functional.normalize(projected, p=2, dim=-1)

    @torch.no_grad()
    def project_all_items(
        self,
        all_embeddings: np.ndarray,
        batch_size: int = 1024,
    ) -> np.ndarray:
        """Project the full item catalogue for FAISS index construction.

        This convenience method runs in inference mode (no gradients) and
        processes items in batches to limit memory usage.

        Parameters
        ----------
        all_embeddings : np.ndarray
            Raw SentenceBERT concatenated embeddings for every movie,
            shape ``(number_of_movies, input_dimension)``.
        batch_size : int
            Number of items to process per forward pass.

        Returns
        -------
        np.ndarray
            L2-normalised projected embeddings, shape
            ``(number_of_movies, projection_dimension)``, dtype float32.
        """
        was_training = self.training
        self.eval()

        device = next(self.parameters()).device
        all_embeddings_tensor = torch.from_numpy(
            all_embeddings.astype(np.float32),
        )

        projected_chunks: list[np.ndarray] = []
        for start in range(0, len(all_embeddings_tensor), batch_size):
            chunk = all_embeddings_tensor[start : start + batch_size].to(device)
            projected = self.forward(chunk)
            projected_chunks.append(projected.cpu().numpy())

        if was_training:
            self.train()

        return np.concatenate(projected_chunks, axis=0).astype(np.float32)
