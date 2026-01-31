from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, kw_only=True)
class Rating:
    movie_id: int
    score: float


class RecommenderModel(ABC):

    @abstractmethod
    def predict(
        self,
        users: pd.DataFrame,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        k: int = 10,
    ) -> np.ndarray:
        """Produce top-K recommendations for each user.

        Parameters
        ----------
        users : pd.DataFrame
            User side-information (UserID, Gender, Age, Occupation, Zip-code).
        ratings : pd.DataFrame
            Observed interactions (UserID, MovieID, Rating, Timestamp).
        movies : pd.DataFrame
            Movie side-information (MovieID, Title, Genres).
        k : int
            Number of recommendations per user.

        Returns
        -------
        np.ndarray
            Array of Rating objects ordered by score (descending),
            shape (n_users, k).
        """
        ...
