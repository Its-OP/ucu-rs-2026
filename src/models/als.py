from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import sparse

from .base import RecommenderModel, Rating


class ALSRecommender(RecommenderModel):
    """Alternating Least Squares matrix factorization for explicit feedback.
    
    Minimizes: Σ_{observed} (r_ui - x_u · y_i)² + λ(||X||² + ||Y||²)
    
    Parameters
    ----------
    n_factors : int
        Dimensionality of latent factors.
    n_iterations : int
        Number of ALS iterations.
    regularization : float
        L2 regularization strength.
    random_state : int
        Seed for reproducibility.
    """
    
    def __init__(
        self,
        n_factors: int = 50,
        n_iterations: int = 15,
        regularization: float = 0.1,
        random_state: int = 42,
    ):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.random_state = random_state
        
        self.global_mean: float = 0.0
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.user_to_idx: Dict[int, int] = {}
        self.item_to_idx: Dict[int, int] = {}
        self.idx_to_item: Dict[int, int] = {}
        
        self.loss_history_: List[float] = []
        
    def _build_interaction_matrix(self, ratings: pd.DataFrame) -> sparse.csr_matrix:
        users_list = ratings["UserID"].unique()
        items_list = ratings["MovieID"].unique()
        
        self.user_to_idx = {u: i for i, u in enumerate(users_list)}
        self.item_to_idx = {m: i for i, m in enumerate(items_list)}
        self.idx_to_item = {i: m for m, i in self.item_to_idx.items()}
        
        row = ratings["UserID"].map(self.user_to_idx).values
        col = ratings["MovieID"].map(self.item_to_idx).values
        data = ratings["Rating"].values
        
        return sparse.csr_matrix(
            (data, (row, col)),
            shape=(len(users_list), len(items_list)),
            dtype=np.float32
        )
    
    def fit(self, ratings: pd.DataFrame) -> "ALSRecommender":
        np.random.seed(self.random_state)
        
        R = self._build_interaction_matrix(ratings)
        n_users, n_items = R.shape
        
        self.global_mean = ratings["Rating"].mean() # center ratings by global mean
        R_centered = R.copy()
        R_centered.data = R_centered.data - self.global_mean
        
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors)).astype(np.float32)
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors)).astype(np.float32)
        
        reg_I = self.regularization * np.eye(self.n_factors, dtype=np.float32)
        
        # alternating
        for iteration in range(self.n_iterations):
            self._als_step_explicit(R_centered, self.item_factors, self.user_factors, reg_I) # fix items, solve for users
            self._als_step_explicit(R_centered.T.tocsr(), self.user_factors, self.item_factors, reg_I) # fix users, solve for items
            
            pred = self.user_factors @ self.item_factors.T
            mask = R.toarray() > 0
            diff = (R.toarray() - self.global_mean - pred) * mask
            rmse = np.sqrt(np.sum(diff ** 2) / np.sum(mask))
            self.loss_history_.append(rmse)
            
            if (iteration + 1) % 5 == 0:
                print(f"  ALS iteration {iteration + 1}/{self.n_iterations}, RMSE: {rmse:.4f}")
        
        return self
    
    def _als_step_explicit(
        self,
        R: sparse.csr_matrix,
        fixed: np.ndarray,
        solve_for: np.ndarray,
        reg_I: np.ndarray,
    ) -> None:
        for u in range(solve_for.shape[0]):
            start, end = R.indptr[u], R.indptr[u + 1]
            indices = R.indices[start:end]
            
            if len(indices) == 0:
                continue
            
            r_u = R.data[start:end] # centered ratings
            Y_u = fixed[indices] # item factors for rated items
            
            # (Y^T Y + λI)^{-1} Y^T r
            A = Y_u.T @ Y_u + reg_I
            b = Y_u.T @ r_u
            
            solve_for[u] = np.linalg.solve(A, b)
    
    def predict(
        self,
        users: pd.DataFrame,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        k: int = 10,
    ) -> Dict[int, List[Rating]]:
        ratings_by_user = {uid: set(grp["MovieID"].values) for uid, grp in ratings.groupby("UserID")}
        
        preds: Dict[int, List[Rating]] = {}
        
        for uid in users["UserID"].values:
            uid = int(uid)
            seen = ratings_by_user.get(uid, set())
            
            if uid not in self.user_to_idx:
                preds[uid] = []
                continue
            
            u_idx = self.user_to_idx[uid]
            
            # Predicted rating = global_mean + x_u · y_i
            scores = self.global_mean + self.user_factors[u_idx] @ self.item_factors.T
            
            # Mask seen items
            for mid in seen:
                if mid in self.item_to_idx:
                    scores[self.item_to_idx[mid]] = -np.inf
            
            # Top-k
            top_indices = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
            
            preds[uid] = [
                Rating(movie_id=int(self.idx_to_item[i]), score=float(scores[i]))
                for i in top_indices
                if scores[i] > -np.inf
            ]
        
        return preds
      
      
