from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).parent

movies = pd.read_csv(f'{_DATA_DIR}/movies.dat',
                     sep='::',
                     engine='python',
                     names=['MovieID', 'Title', 'Genres'],
                     encoding='latin-1')

users = pd.read_csv(f'{_DATA_DIR}/users.dat',
                    sep='::',
                    engine='python',
                    names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'],
                    encoding='latin-1')

ratings = pd.read_csv(f'{_DATA_DIR}/ratings.dat',
                      sep='::',
                      engine='python',
                      names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
                      encoding='latin-1')
ratings['Timestamp'] = pd.to_datetime(ratings['Timestamp'], unit='s')

# Temporal train / validation / test split  (80 / 10 / 10 by timestamp)

_t1 = ratings['Timestamp'].quantile(0.8)
_t2 = ratings['Timestamp'].quantile(0.9)

train = ratings[ratings['Timestamp'] < _t1].copy()
val = ratings[(ratings['Timestamp'] >= _t1) & (ratings['Timestamp'] < _t2)].copy()
test = ratings[ratings['Timestamp'] >= _t2].copy()