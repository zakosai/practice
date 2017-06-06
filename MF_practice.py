__author__ = 'linh'

import pandas as pd
from MF import MF
import time
from sklearn.model_selection import train_test_split

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

# ratings_base = pd.read_csv('ml-100k/ub.base', sep='\t', names=r_cols, encoding='latin-1')
# ratings_test = pd.read_csv('ml-100k/ub.test', sep='\t', names=r_cols, encoding='latin-1')
#
# rate_train = ratings_base.as_matrix()
# rate_test = ratings_test.as_matrix()
#
# # indices start from 0
# rate_train[:, :2] -= 1
# rate_test[:, :2] -= 1

ratings_based = pd.read_csv('ml-10M100K/ratings.dat', sep='::', names=r_cols, encoding='latin-1')
ratings = ratings_based.as_matrix()

ratings[:, :2] -= 1
rate_train, rate_test = train_test_split(ratings, test_size=0.33, random_state=42)
print(rate_train.shape, rate_test.shape)

start = time.time()
rs = MF(rate_train, K = 10, lam = .1, print_every = 10,
    learning_rate = 0.75, max_iter = 50, user_based = 1)
rs.fit()
print(time.time() - start)

start = time.time()
# evaluate on test data
RMSE = rs.evaluate_RMSE(rate_test)
print('\nUser-based MF, RMSE =', RMSE)
print(time.time()-start)