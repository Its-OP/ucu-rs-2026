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

# Per-user temporal train / val / test split (80 / 10 / 10)

_train, _val, _test = [], [], []

for _, _group in ratings.groupby('UserID'):
    _group = _group.sort_values('Timestamp')
    _n = len(_group)
    _t1 = int(_n * 0.75)
    _t2 = int(_n * 0.99)

    _train.append(_group.iloc[:_t1])
    _val.append(_group.iloc[_t1:_t2])
    _test.append(_group.iloc[_t2:])

train = pd.concat(_train, ignore_index=True)
val = pd.concat(_val, ignore_index=True)
test = pd.concat(_test, ignore_index=True)