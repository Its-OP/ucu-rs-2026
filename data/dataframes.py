from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

_DATA_DIR = f'{Path(__file__).parent}/datasets'

PCA_DIM = 400

movies = pd.read_csv(f'{_DATA_DIR}/movies.dat',
                     sep='::',
                     engine='python',
                     names=['MovieID', 'Title', 'Genres'],
                     encoding='latin-1')

movies_enriched = pd.read_csv(f'{_DATA_DIR}/movies_enriched.csv')
embeddings = np.load(f'{_DATA_DIR}/embeddings.npz')
_raw = embeddings['concat']
_reduced = PCA(n_components=PCA_DIM).fit_transform(_raw).astype('float32')
movies_enriched['embedding'] = list(_reduced)

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

# Temporal train / val / test split  (75 / 12.5 / 12.5 by timestamp)

_t1 = ratings["Timestamp"].quantile(0.75)
_t2 = ratings["Timestamp"].quantile(0.875)

train = ratings[ratings['Timestamp'] < _t1].copy()
val = ratings[(ratings['Timestamp'] >= _t1) & (ratings['Timestamp'] < _t2)].copy()
test = ratings[ratings['Timestamp'] >= _t2].copy()
