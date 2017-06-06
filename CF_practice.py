__author__ = 'linh'
import pandas as pd
from CF import CF
import numpy as np
import time

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('ml-100k/ub.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ub.test', sep='\t', names=r_cols, encoding='latin-1')

rate_train = ratings_base.as_matrix()
rate_test = ratings_test.as_matrix()

# indices start from 0
rate_train[:, :2] -= 1
rate_test[:, :2] -= 1

start = time.time()
rs = CF(rate_train, k = 10, uuCF = 0)
rs.fit()
print(time.time() - start)

start = time.time()
n_tests = rate_test.shape[0]
SE = 0 # squared error
for n in range(0,n_tests):
    pred = rs.pred(rate_test[n, 0], rate_test[n, 1], normalized = 0)
    SE += (pred - rate_test[n, 2])**2

RMSE = np.sqrt(SE/n_tests)
print('item-item CF, RMSE =', RMSE)
print(time.time()-start)
