from __future__ import annotations
from typing import Dict, List

import numpy as np
import pandas as pd

from .base import RecommenderModel, Rating


class FunkSVDRecommender(RecommenderModel):
    """FunkSVD matrix factorization using stochastic gradient descent.
    
    Parameters
    ----------
    n_factors : int
        Dimensionality of latent factors.
    n_epochs : int
        Number of training epochs.
    lr : float
        Learning rate for SGD.
    regularization : float
        L2 regularization strength.
    random_state : int
        Seed for reproducibility.
    """
    
    def __init__(
        self,
        n_factors: int = 50,
        n_epochs: int = 20,
        lr: float = 0.005,
        regularization: float = 0.02,
        random_state: int = 42,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.regularization = regularization
        self.random_state = random_state
        
        self.global_mean: float = 0.0
        self.user_bias: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        
        self.user_to_idx: Dict[int, int] = {}
        self.item_to_idx: Dict[int, int] = {}
        self.idx_to_item: Dict[int, int] = {}
        
        self.loss_history_: List[float] = []
    
    def fit(self, ratings: pd.DataFrame) -> "FunkSVDRecommender":
        np.random.seed(self.random_state)
        
        # mappings
        users_list = ratings["UserID"].unique()
        items_list = ratings["MovieID"].unique()
        
        self.user_to_idx = {u: i for i, u in enumerate(users_list)}
        self.item_to_idx = {m: i for i, m in enumerate(items_list)}
        self.idx_to_item = {i: m for m, i in self.item_to_idx.items()}
        
        n_users = len(users_list)
        n_items = len(items_list)
        
        # params
        self.global_mean = ratings["Rating"].mean()
        self.user_bias = np.zeros(n_users, dtype=np.float32)
        self.item_bias = np.zeros(n_items, dtype=np.float32)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors)).astype(np.float32)
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors)).astype(np.float32)
        
        train_data = ratings[["UserID", "MovieID", "Rating"]].values
        
        # train
        for epoch in range(self.n_epochs):
            np.random.shuffle(train_data)
            total_loss = 0.0
            
            for uid, mid, rating in train_data:
                u_idx = self.user_to_idx[uid]
                i_idx = self.item_to_idx[mid]
                
                # predict
                pred = (
                    self.global_mean 
                    + self.user_bias[u_idx] 
                    + self.item_bias[i_idx]
                    + self.user_factors[u_idx] @ self.item_factors[i_idx]
                )
                
                # loss
                err = rating - pred
                total_loss += err ** 2
                
                # biases/factors update
                self.user_bias[u_idx] += self.lr * (err - self.regularization * self.user_bias[u_idx])
                self.item_bias[i_idx] += self.lr * (err - self.regularization * self.item_bias[i_idx])
                
                u_factors_old = self.user_factors[u_idx].copy()
                self.user_factors[u_idx] += self.lr * (err * self.item_factors[i_idx] - self.regularization * self.user_factors[u_idx])
                self.item_factors[i_idx] += self.lr * (err * u_factors_old - self.regularization * self.item_factors[i_idx])
            
            rmse = np.sqrt(total_loss / len(train_data))
            self.loss_history_.append(rmse)
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch + 1}/{self.n_epochs}, RMSE: {rmse:.4f}")
        
        return self
    
    def _predict_score(self, u_idx: int, i_idx: int) -> float:
        return (
            self.global_mean
            + self.user_bias[u_idx]
            + self.item_bias[i_idx]
            + self.user_factors[u_idx] @ self.item_factors[i_idx]
        )
    
    def predict(
        self,
        users: pd.DataFrame,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        k: int = 10,
    ) -> Dict[int, List[Rating]]:
        ratings_by_user = {uid: set(grp["MovieID"].values) for uid, grp in ratings.groupby("UserID")}
        
        preds: Dict[int, List[Rating]] = {}
        n_items = len(self.item_to_idx)
        
        for uid in users["UserID"].values:
            uid = int(uid)
            seen = ratings_by_user.get(uid, set())
            
            if uid not in self.user_to_idx:
                preds[uid] = []
                continue
            
            u_idx = self.user_to_idx[uid]
            
            # scores for all items
            scores = (
                self.global_mean
                + self.user_bias[u_idx]
                + self.item_bias
                + self.item_factors @ self.user_factors[u_idx]
            )
            
            # mask seen items
            for mid in seen:
                if mid in self.item_to_idx:
                    scores[self.item_to_idx[mid]] = -np.inf
            
            # top-k
            top_indices = np.argpartition(-scores, min(k, n_items - 1))[:k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
            
            preds[uid] = [
                Rating(movie_id=int(self.idx_to_item[i]), score=float(scores[i]))
                for i in top_indices
                if scores[i] > -np.inf
            ]
        
        return preds
      
