from __future__ import annotations
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import sparse

from .base import RecommenderModel, Rating


def cosine_similarity_matrix(R: sparse.csr_matrix) -> np.ndarray:
    R_csc = R.tocsc().astype(np.float32)
    
    col_sq = np.array(R_csc.power(2).sum(axis=0)).ravel() # fast column norms sqrt(sum of squares)
    norms = np.sqrt(col_sq)
    norms[norms == 0] = 1.0
    
    R_norm = R_csc.multiply(1.0 / norms)
    
    S = (R_norm.T @ R_norm).toarray().astype(np.float32)
    return S


def adjusted_cosine_similarity_matrix(R: sparse.csr_matrix) -> np.ndarray:
    R_dense = R.toarray().astype(np.float32)
    
    # user means
    user_sums = R.sum(axis=1).A1
    user_counts = (R > 0).sum(axis=1).A1
    user_counts[user_counts == 0] = 1
    user_means = (user_sums / user_counts).astype(np.float32)
    
    # center by user mean
    mask = (R_dense > 0).astype(np.float32)
    R_centered = R_dense - (user_means[:, None] * mask)
    
    # cosine on centered
    norms = np.linalg.norm(R_centered, axis=0)
    norms[norms == 0] = 1
    R_norm = R_centered / norms
    
    S = (R_norm.T @ R_norm).astype(np.float32)
    return S


def pearson_similarity_matrix(R: sparse.csr_matrix) -> np.ndarray:
    R_dense = R.toarray().astype(np.float32)
    
    # item means
    item_sums = R.sum(axis=0).A1
    item_counts = (R > 0).sum(axis=0).A1
    item_counts[item_counts == 0] = 1
    item_means = (item_sums / item_counts).astype(np.float32)
    
    # center by item mean
    mask = (R_dense > 0).astype(np.float32)
    R_centered = R_dense - (item_means * mask)
    
    # cosine on centered
    norms = np.linalg.norm(R_centered, axis=0)
    norms[norms == 0] = 1
    R_norm = R_centered / norms
    
    S = (R_norm.T @ R_norm).astype(np.float32)
    return S


SIMILARITY_FUNCTIONS = {
    "cosine": cosine_similarity_matrix,
    "adjusted_cosine": adjusted_cosine_similarity_matrix,
    "pearson": pearson_similarity_matrix,
}


class ItemItemCF(RecommenderModel):
    """Item-Item Collaborative Filtering with exact rating prediction.
    
    Predicts ratings on 1-5 scale using weighted average of similar items.
    Fully vectorized for speed.
    
    Parameters
    ----------
    similarity : str
        Similarity function: 'cosine', 'adjusted_cosine', or 'pearson'.
    k_neighbors : int
        Number of most similar items to use for prediction.
    """
    
    def __init__(
        self,
        similarity: str = "cosine",
        k_neighbors: int = 50,
    ):
        self.similarity = similarity
        self.k_neighbors = k_neighbors
        
        self.sim_matrix: np.ndarray | None = None
        self.R: np.ndarray | None = None
        self.global_mean: float = 3.0
        self.item_means: np.ndarray | None = None
        
        self.item_to_idx: Dict[int, int] = {}
        self.idx_to_item: Dict[int, int] = {}
        self.user_to_idx: Dict[int, int] = {}
        
    def fit(self, ratings: pd.DataFrame) -> "ItemItemCF":
        """Fit model by computing item-item similarity matrix."""
        users_list = ratings["UserID"].unique()
        items_list = ratings["MovieID"].unique()
        
        self.user_to_idx = {u: i for i, u in enumerate(users_list)}
        self.item_to_idx = {m: i for i, m in enumerate(items_list)}
        self.idx_to_item = {i: m for m, i in self.item_to_idx.items()}
        
        # sparse rating matrix
        row = ratings["UserID"].map(self.user_to_idx).values
        col = ratings["MovieID"].map(self.item_to_idx).values
        data = ratings["Rating"].values.astype(np.float32)
        
        R_sparse = sparse.csr_matrix(
            (data, (row, col)),
            shape=(len(users_list), len(items_list)),
            dtype=np.float32
        )
        
        self.R = R_sparse.toarray()
        
        # fallback
        self.global_mean = float(ratings["Rating"].mean())
        item_sums = R_sparse.sum(axis=0).A1
        item_counts = (R_sparse > 0).sum(axis=0).A1
        item_counts[item_counts == 0] = 1
        self.item_means = item_sums / item_counts
        
        # similarity matrix
        sim_func = SIMILARITY_FUNCTIONS[self.similarity]
        print(f"  Computing {self.similarity} similarity matrix...")
        self.sim_matrix = sim_func(R_sparse)
        
        np.fill_diagonal(self.sim_matrix, 0) # zero out self-similarity
        
        # top-k neighbors for each item pre-computed for speed optimization
        print(f"  Pre-computing top-{self.k_neighbors} neighbors...")
        n_items = self.sim_matrix.shape[0]
        self.top_k_neighbors = np.zeros((n_items, self.k_neighbors), dtype=np.int32)
        self.top_k_sims = np.zeros((n_items, self.k_neighbors), dtype=np.float32)
        
        for i in range(n_items):
            sims = self.sim_matrix[i]
            # top-k by absolute similarity
            if n_items > self.k_neighbors:
                top_idx = np.argpartition(-np.abs(sims), self.k_neighbors)[:self.k_neighbors]
            else:
                top_idx = np.arange(n_items)
            # sort by similarity descending
            sorted_order = np.argsort(-np.abs(sims[top_idx]))
            top_idx = top_idx[sorted_order[:self.k_neighbors]]
            
            self.top_k_neighbors[i, :len(top_idx)] = top_idx
            self.top_k_sims[i, :len(top_idx)] = sims[top_idx]
        
        return self
    
    def _predict_scores_for_user_vectorized(self, u_idx: int) -> np.ndarray:
        user_ratings = self.R[u_idx] # (n_items,)
        rated_mask = user_ratings > 0
        n_items = len(user_ratings)
        
        if not rated_mask.any():
            return np.full(n_items, self.global_mean, dtype=np.float32)
        
        # ratings of neighbors
        neighbor_ratings = user_ratings[self.top_k_neighbors] # (n_items, k)
        neighbor_sims = self.top_k_sims.copy() # (n_items, k)
        
        # mask unrated neighbors
        neighbor_rated = neighbor_ratings > 0
        neighbor_sims_masked = neighbor_sims * neighbor_rated
        
        # weighted sum scores
        numerator = np.sum(neighbor_sims_masked * neighbor_ratings, axis=1)
        denominator = np.sum(np.abs(neighbor_sims_masked), axis=1)
        
        scores = np.where(
            denominator > 0,
            numerator / denominator,
            self.item_means # fallback to item mean
        )
        
        scores = np.clip(scores, 1.0, 5.0)
        
        return scores.astype(np.float32)
    
    def predict(
        self,
        users: pd.DataFrame,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        k: int = 10,
    ) -> Dict[int, List[Rating]]:
        ratings_by_user = {
            uid: set(grp["MovieID"].values) 
            for uid, grp in ratings.groupby("UserID")
        }
        
        preds: Dict[int, List[Rating]] = {}
        
        for uid in users["UserID"].values:
            uid = int(uid)
            seen = ratings_by_user.get(uid, set())
            
            if uid not in self.user_to_idx:
                preds[uid] = []
                continue
            
            u_idx = self.user_to_idx[uid]
            scores = self._predict_scores_for_user_vectorized(u_idx)
            
            # mask seen items
            scores_for_ranking = scores.copy()
            for mid in seen:
                if mid in self.item_to_idx:
                    scores_for_ranking[self.item_to_idx[mid]] = -np.inf
            
            # top-k indices
            n_items = len(scores_for_ranking)
            valid_k = min(k, (scores_for_ranking > -np.inf).sum())
            if valid_k == 0:
                preds[uid] = []
                continue
                
            top_indices = np.argpartition(-scores_for_ranking, valid_k - 1)[:valid_k]
            top_indices = top_indices[np.argsort(-scores_for_ranking[top_indices])]
            
            preds[uid] = [
                Rating(
                    movie_id=int(self.idx_to_item[i]), 
                    score=float(scores[i])
                )
                for i in top_indices
                if scores_for_ranking[i] > -np.inf
            ]
        
        return preds
