import numpy as np
import pandas as pd
import pickle
import logging

from application.config import SAVE_PATH


class My_Rec_Model:
    def __init__(self, logger=None):
        if logger is None:
            self.logger = logging.getLogger()
        else:
            self.logger = logger

    def warmup(self, checkpoint_path=SAVE_PATH):
        with open(checkpoint_path + 'recsys_checkpoint.pickle', 'rb') as f:
            self = pickle.load(f)

        self.logger.info(f"Model warmed up from {checkpoint_path}")
        return

    def find_similar(self, movie_id, top_k):
        movie_index = self.movie_id_map[movie_id]

        movie_similarities = np.dot(self.item_factors[movie_index], self.item_factors.T)
        movie_similarities[movie_index] = -np.inf  # minimize similarity with itself

        movie_names, similarity_values = self._get_top_k_similar_movies(movie_similarities, top_k)
        movie_names, similarity_values = self.sort_recommendatdions(movie_names, similarity_values)

        self.logger.info(f"Found {top_k} similar movies: {movie_names}")
        return movie_names, similarity_values

    def predict(self, reviews, top_m):
        pass

    # user_reviews = self._create_user_review(reviews)
    # user_similarities = np.matmul(user_reviews, self.ratings.T)
    # user_similarities = np.expand_dims(user_similarities, axis=0)
    # user_scores = self._get_scores(self.ratings, user_similarities)[0]
    #
    # scores, indexes = self._get_top_k(user_scores, top_m)
    # ids = self._get_movie_ids(indexes)
    # return self.df_movie[self.df_movie.movie_id.isin(ids)]

    def evaluate(self, dataset_path):
        df_rating_test = pd.read_csv(
            dataset_path, sep='::', header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )
        ratings = self._create_ratings(df_rating_test)

    def train(self, dataset_path, epochs=10, regularization_lambda=0.01, n_latent_factors=100):
        self.logger.info(f"Start training...")

        self._load_dataset(dataset_path)

        ratings = self._create_ratings(self.df_rating)
        self.user_factors, self.item_factors = self._create_factors(n_latent_factors)

        train_loss_history = []
        for _ in range(epochs):
            self.user_factors = self._train_step(self.item_factors, ratings, n_latent_factors, regularization_lambda)
            self.item_factors = self._train_step(self.user_factors, ratings.T, n_latent_factors, regularization_lambda)

            predictions = self.predict_ratings()
            train_loss_history.append(
                self.compute_rmse(ratings, predictions)
            )

        self._save(SAVE_PATH)
        self.logger.info(f"Train finished. RMSE: {train_loss_history}")
        return train_loss_history

    def predict_ratings(self):
        return (np.dot(
            self.user_factors,
            self.item_factors.T
        ) * self.user_ratings_mean.reshape(-1, 1)) + self.user_ratings_mean.reshape(-1, 1)

    def _create_ratings(self, df_rating):
        ratings = np.zeros((self.n_users, self.n_items))
        for row in df_rating.itertuples(index=False):
            ratings[row.user_id, row.movie_id] = row.rating
        ratings[ratings == 0] = np.nan
        self.user_ratings_mean = np.nanmean(ratings, axis=1)
        ratings = (ratings - self.user_ratings_mean.reshape(-1, 1)) / self.user_ratings_mean.reshape(-1, 1)
        # ratings[np.isnan(ratings)] = 0
        return ratings

    def _get_top_k_similar_movies(self, movie_similarities, top_k):
        top_similarities = np.argpartition(movie_similarities, -top_k)[-top_k:]
        similarity_values = movie_similarities[top_similarities].tolist()

        movie_ids = list(map(
            lambda x: self.movie_id_map_inverse[x],
            top_similarities
        ))
        movie_names = self.df_movie[self.df_movie.movie_id.isin(movie_ids)].title.tolist()
        return movie_names, similarity_values

    def _train_step(self, fix_matrix, ratings, n_latent_factors, regularization_lambda):
        return np.dot(
            np.dot(ratings, fix_matrix),
            np.linalg.inv(
                np.dot(fix_matrix.T, fix_matrix) + np.eye(n_latent_factors) * regularization_lambda
            )
        )

    def _load_dataset(self, dataset_path):
        rating_path = dataset_path + '/ratings_train.dat'
        self.df_rating = pd.read_csv(
            rating_path, sep='::', header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )
        self.df_rating.movie_id = self.df_rating.movie_id.apply(lambda x: f'mid-{x}')
        self.df_rating.user_id = self.df_rating.user_id.apply(lambda x: f'uid-{x}')

        self.user_id_map, self.user_id_map_inverse = self._get_map(self.df_rating['user_id'].unique())
        self.movie_id_map, self.movie_id_map_inverse = self._get_map(self.df_rating['movie_id'].unique())

        self.n_users, self.n_items = len(self.user_id_map), len(self.movie_id_map)

        self.df_rating['movie_id_init'] = self.df_rating.movie_id
        self.df_rating['user_id_init'] = self.df_rating.user_id

        self.df_rating.movie_id = self.df_rating.movie_id.apply(lambda x: self.movie_id_map[x])
        self.df_rating.user_id = self.df_rating.user_id.apply(lambda x: self.user_id_map[x])

        # user_path = dataset_path + '/users.dat'
        # self.df_user = pd.read_csv(
        #     user_path, sep='::', header=None,
        #     names=['user_id', 'gender', 'age', 'occupation', 'zipcode']
        # )

        movie_path = dataset_path + '/movies.dat'
        self.df_movie = pd.read_csv(
            movie_path, sep='::', header=None,
            names=['movie_id', 'title', 'genre'],
            encoding='windows-1251'
        )
        self.df_movie.movie_id = self.df_movie.movie_id.apply(lambda x: f'mid-{x}')

        self._movie_name_to_id = {r.title: r.movie_id for _, r in self.df_movie.iterrows()}
        self._movie_id_to_name = {movie_id: title for title, movie_id in self._movie_name_to_id.items()}

    def movie_name_to_id(self, name):
        return self._movie_name_to_id[name]

    def movie_id_to_name(self, id):
        return self._movie_id_to_name[id]

    def _create_factors(self, num_factors):
        user_factors = np.random.random((self.n_users, num_factors))
        item_factors = np.random.random((self.n_items, num_factors))
        return user_factors, item_factors

    def _save(self, save_path):
        with open(save_path + 'recsys_checkpoint.pickle', 'wb') as f:
            pickle.dump(self, f)

        self.logger.info(f"Model saved to {save_path}\n")

    def _get_map(self, unique_ids):
        id_map = {id: idx for idx, id in enumerate(unique_ids)}
        id_map_inverse = {idx: id for id, idx in id_map.items()}
        return id_map, id_map_inverse

    @staticmethod
    def compute_rmse(y_true, y_pred):
        rmse = np.sqrt(
            np.mean((y_true - y_pred) ** 2)
        )
        return rmse

    @staticmethod
    def sort_recommendatdions(movie_names, similarity_values):
        recommendations = sorted(zip(movie_names, similarity_values), key=lambda x: x[1], reverse=True)
        recommendations = list(zip(*recommendations))
        return recommendations[0], recommendations[1]
