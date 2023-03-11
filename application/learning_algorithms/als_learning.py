import numpy as np


class ALSLearning:
    def __init__(self, predict_ratings, epochs=10, regularization_lambda=0.01, n_latent_factors=30):
        self.epochs = epochs
        self.regularization_lambda = regularization_lambda
        self.n_latent_factors = n_latent_factors
        self.predict_ratings = predict_ratings

    def fit(self, ratings):
        user_factors, item_factors = self._create_factors(ratings.shape[0], ratings.shape[1], self.n_latent_factors)

        for _ in range(self.epochs):
            user_factors = self._train_step(item_factors, ratings, self.n_latent_factors, self.regularization_lambda)
            item_factors = self._train_step(user_factors, ratings.T, self.n_latent_factors, self.regularization_lambda)

        return user_factors, item_factors

    def _train_step(self, fix_matrix, ratings, n_latent_factors, regularization_lambda):
        return np.dot(
            np.dot(ratings, fix_matrix),
            np.linalg.inv(
                np.dot(fix_matrix.T, fix_matrix) + np.eye(n_latent_factors) * regularization_lambda
            )
        )

    def _create_factors(self, n_users, n_items, num_factors):
        user_factors = np.random.random((n_users, num_factors))
        item_factors = np.random.random((n_items, num_factors))
        return user_factors, item_factors
