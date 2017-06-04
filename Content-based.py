__author__ = 'linh'

import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.linear_model import Ridge
from sklearn import linear_model


def getDataInfo():
    #Reading users file:
    u_cols =  ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')

    n_users = users.shape[0]
    print('Number of users:', n_users)

    #Reading ratings file:
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

    ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
    ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')

    rate_train = ratings_base.as_matrix()
    rate_test = ratings_test.as_matrix()

    print('Number of traing rates:', rate_train.shape[0])
    print('Number of test rates:', rate_test.shape[0])

    #Reading items file:
    i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
     'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')

    n_items = items.shape[0]
    print('Number of items:', n_items)

    X0 = items.as_matrix()
    X_train_counts = X0[:,-19:]
    # for i in range(0,5):
    #     print(X0[i])

    for i in range(0, 5):
        l = []
        for j in range(19, 0, -1):
            if int(X0[i, -j]) == 1:
                l.append(i_cols[-j])

        print(X0[i,1], ':', l)

    return X0, X_train_counts, rate_train, n_users, rate_test

def get_items_rated_by_user(rate_matrix, user_id):
    y = rate_matrix[:,0]
    ids = np.where(y == user_id + 1)[0]
    item_ids = rate_matrix[ids, 1] - 1
    scores = rate_matrix[ids, 2]
    return item_ids, scores

def evaluate(Yhat, rates, W, b, n_users):
    se = 0
    cnt = 0
    for n in range(0, n_users):
        ids, scores_truth = get_items_rated_by_user(rates, n)
        scores_pred = Yhat[ids, n]
        e = scores_truth - scores_pred
        se += (e*e).sum(axis = 0)
        cnt += e.size
    return np.sqrt(se/cnt)

def main():
    X0, X_train_counts, rate_train, n_users, rate_test = getDataInfo()
    transformer = TfidfTransformer(smooth_idf=True, norm='l2')
    tfidf = transformer.fit_transform(X_train_counts.tolist()).toarray()

    d = tfidf.shape[1]
    W = np.zeros((d, n_users))
    b = np.zeros((1, n_users))

    for n in range(1, n_users):
        ids, scores = get_items_rated_by_user(rate_train, n)
        clf = Ridge(alpha=0.01, fit_intercept=True)
        Xhat = tfidf[ids, :]

        clf.fit(Xhat, scores)
        W[:,n] = clf.coef_
        b[0,n] = clf.intercept_

    Yhat = tfidf.dot(W) + b

    print('RMSE for training:', evaluate(Yhat, rate_train, W, b, n_users))
    print('RMSE for test    :', evaluate(Yhat, rate_test, W, b, n_users))

    #test with 1 user
    n = 10
    ids, scores = get_items_rated_by_user(rate_train, n)
    print('Rated movies ids:', ids)
    print('True ratings:', scores)
    print('Predicted ratings:', Yhat[ids, n])
    i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
     'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    ids = []
    for i in range(0, 1682):
        if i not in ids and Yhat[i, n]  >= 5:
            ids.append(i)

    print(ids)

    for i in ids:
        l = []
        for j in range(19, 0, -1):
            if int(X0[i, -j]) == 1:
                l.append(i_cols[-j])

        print(X0[i,1], ':', l)



if __name__ == '__main__':
    main()







