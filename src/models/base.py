from abc import ABC, abstractmethod
from dataclasses import dataclass

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
    ) -> dict[int, list[Rating]]:
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
        dict[int, list[Rating]]
            Mapping from UserID to a list of Rating objects sorted by
            score descending (length up to k).
        """
        ...
