__author__ = 'linh'

import numpy as np


class MF(object):
    def __init__(self, Y_data, K, lam = 0.1, Xinit = None, Winit = None, learning_rate = 0.5,
                 max_iter = 1000, print_every = 100, user_based = 1 ):
        self.Y_raw_data = Y_data
        self.K = K #dimension of X and W
        self.lam = lam
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.print_every = print_every
        self.user_based = user_based

        self.n_users = int(np.max(Y_data[:,0])) + 1
        self.n_items = int(np.max(Y_data[:,1])) + 1
        self.n_ratings = Y_data.shape[0]

        if Xinit is None:
            self.X = np.random.randn(self.n_items, K)
        else:
            self.X = Xinit

        if Winit is None:
            self.W = np.random.randn(K, self.n_users)
        else:
            self.W = Winit

        #normalized data
        self.Y_data_n = self.Y_raw_data.copy()


    def normalize_Y(self):
        if self.user_based:
            user_col = 1
            item_col = 0
            n_objects = self.n_users
        else:
            user_col = 0
            item_col = 1
            n_objects = self.n_items

        users = self.Y_raw_data[:, user_col]
        self.mu = np.zeros((n_objects, ))
        for n in range(n_objects):
            ids = np.where(users == n)[0].astype(np.int32)
            item_ids = self.Y_data_n[ids, item_col]
            ratings = self.Y_data_n[ids, 2]

            m = np.mean(ratings)
            if np.isnan(m):
                m = 0
            self.mu[n] = m
            self.Y_data_n[ids, 2] = ratings - self.mu[n]


    def loss(self):
        L = 0
        for i in range(self.n_ratings):
            n, m , rating = int(self.Y_data_n[i, 0]), int(self.Y_data_n[i,1]), self.Y_data_n[i, 2]
            L += 0.5*(rating - self.X[m, :].dot(self.W[:,n]))**2

        # take average
        L /= self.n_ratings
        L += 0.5*self.lam*(np.linalg.norm(self.X, 'fro') + np.linalg.norm(self.W, 'fro'))

        return L


    def get_item_rated_by_user(self, user_id):
        ids = np.where(self.Y_data_n[:, 0] == user_id)[0]
        items = self.Y_data_n[ids, 1].astype(np.int32)
        ratings = self.Y_data_n[ids, 2]

        return items, ratings


    def get_users_who_rate_item(self, item_id):
        ids = np.where(self.Y_data_n[:, 1] == item_id)[0]
        users = self.Y_data_n[ids, 0].astype(np.int32)
        ratings = self.Y_data_n[ids, 2]

        return users, ratings


    def updateX(self):
        for m in range(self.n_items):
            user_ids, ratings = self.get_users_who_rate_item(m)
            Wm = self.W[:, user_ids]
            grad_xm = -(ratings - self.X[m, :].dot(Wm)).dot(Wm.T)/self.n_ratings + self.lam*(self.X[m, :])

            self.X[m, :] -= self.learning_rate*grad_xm.reshape((self.K, ))


    def updateW(self):
        for n in range(self.n_users):
            item_ids, ratings = self.get_item_rated_by_user(n)
            Xn = self.X[item_ids, :]
            grad_wn = -Xn.T.dot(ratings - Xn.dot(self.W[:, n]))/self.n_ratings + self.lam*(self.W[:, n])

            self.W[:, n] -= self.learning_rate*grad_wn.reshape((self.K,))


    def pred(self, u, i):
        u = int(u)
        i = int(i)
        if self.user_based:
            bias = self.mu[u]
        else:
            bias = self.mu[i]

        pred = self.X[i, :].dot(self.W[:, u]) + bias
        if pred < 0:
            return 0
        if pred > 5:
            return 5

        return pred


    def pred_for_user(self, user_id):
        ids = np.where(self.Y_data_n[:,0] == user_id)[0]
        items_rated_by_u = self.Y_data_n[ids, 1].tolist()

        y_pred = self.X.dot(self.W[:, user_id]) + self.mu[user_id]

        predict_ratings = []
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                predict_ratings.append((i, y_pred[i]))
        return predict_ratings


    def evaluate_RMSE(self, rate_test):
        n_test = rate_test.shape[0]
        SE = 0
        for n in range(n_test):
            pred = self.pred(rate_test[n, 0], rate_test[n,1])
            SE += (pred - rate_test[n, 2])**2

        RMSE = np.sqrt(SE/n_test)
        return RMSE


    def fit(self):
        self.normalize_Y()
        for it in range(self.max_iter):
            self.updateX()
            self.updateW()

            if (it+1) % self.print_every == 0:
                rmse_train = self.evaluate_RMSE(self.Y_raw_data)
                print('iter ', it + 1, ', loss = ', self.loss(), ', RMSE train = ' , rmse_train, )








