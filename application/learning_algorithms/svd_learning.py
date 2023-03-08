import numpy as np


class SVDLearning:
    def __init__(self, metric_function, predict_ratings, n_latent_factors):
        self.metric_function = metric_function
        self.predict_ratings = predict_ratings
        self.n_latent_factors = n_latent_factors

    def fit(self, ratings):
        ratings[np.isnan(ratings)] = 0

        U, sigma, V = np.linalg.svd(ratings)  # row-users. col-items
        U = U[:, :self.n_latent_factors]
        V = V.T[:, :self.n_latent_factors]
        sigma = np.diag(sigma[:self.n_latent_factors])

        user_factors = np.dot(U, sigma)
        item_factors = V

        predictions = self.predict_ratings(user_factors, item_factors)
        train_loss_history = [self.metric_function(ratings, predictions)]
        return user_factors, item_factors, train_loss_history
