import numpy as np
import pandas as pd
import pickle
import logging

from application.config import SAVE_PATH
from application.learning_algorithms.als_learning import ALSLearning
from application.learning_algorithms.svd_learning import SVDLearning
from application.utils import timeit


class My_Rec_Model:
    def __init__(self, logger=None):

        self.train_methods = {
            'svd': self.train_svd,
            'als': self.train_als
        }

        if logger is None:
            self.logger = logging.getLogger()
        else:
            self.logger = logger

    def warmup(self, checkpoint_path=SAVE_PATH):
        try:
            with open(checkpoint_path + 'recsys_checkpoint.pickle', 'rb') as f:
                self.__dict__.update(pickle.load(f).__dict__)
        except:
            raise FileNotFoundError(f'No checkpoint for model warmup')

        self.logger.info(f"Model warmed up from {checkpoint_path}")
        return

    @timeit
    def find_similar(self, movie_id, top_k):
        movie_index = self.movie_id_map[movie_id]

        movie_similarities = np.dot(self.item_factors[movie_index], self.item_factors.T)
        movie_similarities[movie_index] = -np.inf  # minimize similarity with itself

        movie_names, similarity_values = self._get_top_k_recommendations(movie_similarities, top_k)
        movie_names, similarity_values = self.sort_recommendatdions(movie_names, similarity_values)

        self.logger.info(f"Found {top_k} similar movies: {movie_names}")
        return movie_names, similarity_values

    @timeit
    def predict(self, user_id, top_m):
        user_index = self.user_id_map[user_id]

        ratings = (np.dot(
            self.user_factors[user_index],
            self.item_factors.T
        ) * self.user_ratings_mean[user_index]) + self.user_ratings_mean[user_index]

        # ratings = self._predict_ratings(
        #     np.expand_dims(self.user_factors[user_index], axis=0),
        #     self.item_factors
        # )
        print(ratings.shape)
        movie_names, similarity_values = self._get_top_k_recommendations(ratings, top_m)
        movie_names, similarity_values = self.sort_recommendatdions(movie_names, similarity_values)
        self.logger.info(f"Recommended {top_m} movies: {movie_names}")
        return movie_names, similarity_values

    def evaluate(self, dataset_path):
        df_rating_test = pd.read_csv(
            dataset_path, sep='::', header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            engine='python'
        )
        df_rating_test.movie_id = df_rating_test.movie_id.apply(lambda x: f'mid-{x}')
        df_rating_test.user_id = df_rating_test.user_id.apply(lambda x: f'uid-{x}')

        df_rating_test['user_id_2'] = df_rating_test.user_id.apply(lambda x: self.user_id_map.get(x))
        df_rating_test['movie_id_2'] = df_rating_test.movie_id.apply(lambda x: self.movie_id_map.get(x))

        ratings = pd.DataFrame(self.predict_ratings())
        preds = pd.melt(ratings.reset_index(), id_vars='index')
        preds = preds.rename(columns={'index': 'user_id_2', 'variable': 'movie_id_2'})
        preds.loc[preds['value'] < 1, 'value'] = 1
        preds.loc[preds['value'] > 5, 'value'] = 5

        mergeres = preds.merge(df_rating_test, on=['user_id_2', 'movie_id_2'])
        rmse_v = self.compute_rmse(mergeres.rating, mergeres.value)
        self.logger.info(f'Evaluated RMSE: {rmse_v}\nDataset: {dataset_path}')
        return rmse_v

    def train(self, dataset_path, method_name, **kwargs):
        self._load_dataset(dataset_path)
        ratings = self._create_ratings(self.df_rating)

        train_method = self.get_train_method(method_name)

        self.user_factors, self.item_factors = train_method(ratings, **kwargs)

        self.evaluate(dataset_path=dataset_path + '/ratings_train.dat')
        self._save(SAVE_PATH)
        self.logger.info(f"Train finished.")

    def train_svd(self, ratings, n_latent_factors=40):
        self.logger.info(f"Start training (SVD)...")
        learning_method = SVDLearning(
            predict_ratings=self._predict_ratings,
            n_latent_factors=n_latent_factors
        )
        return learning_method.fit(ratings)

    def train_als(self, ratings, epochs=10, regularization_lambda=0.01, n_latent_factors=40):
        self.logger.info(f"Start training (ALS)...")
        learning_method = ALSLearning(
            predict_ratings=self._predict_ratings,
            epochs=epochs,
            regularization_lambda=regularization_lambda,
            n_latent_factors=n_latent_factors
        )
        return learning_method.fit(ratings)

    def predict_ratings(self):
        return self._predict_ratings(self.user_factors, self.item_factors)

    def _predict_ratings(self, user_factors, item_factors):
        return (np.dot(
            user_factors,
            item_factors.T
        ) * self.user_ratings_range.reshape(-1, 1)) + self.user_ratings_mean.reshape(-1, 1)

    def get_train_method(self, method_name):
        train_method = self.train_methods.get(method_name)
        if train_method is None:
            raise KeyError(f'No such learning method: {method_name}')
        return train_method

    def _create_ratings(self, df_rating):
        ratings = np.zeros((self.n_users, self.n_items))
        for row in df_rating.itertuples(index=False):
            ratings[row.user_id, row.movie_id] = row.rating
        ratings[ratings == 0] = np.nan

        self.user_ratings_mean = np.nanmean(ratings, axis=1)
        self.user_ratings_range = np.nanmax(ratings, axis=1) - np.nanmin(ratings, axis=1)

        ratings = (ratings - self.user_ratings_mean.reshape(-1, 1)) / self.user_ratings_range.reshape(-1, 1)
        return ratings

    def _get_top_k_recommendations(self, movie_scoers, top_k):
        print('recoms////')
        top_recommendations = np.argpartition(movie_scoers, -top_k)[-top_k:]
        similarity_values = movie_scoers[top_recommendations].tolist()
        print('recoms//// END')
        movie_ids = list(map(
            lambda x: self.movie_id_map_inverse[x],
            top_recommendations
        ))
        movie_names = self.df_movie[self.df_movie.movie_id.isin(movie_ids)].title.tolist()
        return movie_names, similarity_values

    def _load_dataset(self, dataset_path):
        rating_path = dataset_path + '/ratings_train.dat'
        self.df_rating = pd.read_csv(
            rating_path, sep='::', header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            engine='python'
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

        movie_path = dataset_path + '/movies.dat'
        self.df_movie = pd.read_csv(
            movie_path, sep='::', header=None,
            names=['movie_id', 'title', 'genre'],
            encoding='windows-1251',
            engine='python'
        )
        self.df_movie['title'] = self.df_movie.title.str[:-7]
        self.df_movie.movie_id = self.df_movie.movie_id.apply(lambda x: f'mid-{x}')

        self._movie_name_to_id = {r.title: r.movie_id for _, r in self.df_movie.iterrows()}
        self._movie_id_to_name = {movie_id: title for title, movie_id in self._movie_name_to_id.items()}

    def movie_name_to_id(self, name):
        return self._movie_name_to_id[name]

    def movie_id_to_name(self, id):
        return self._movie_id_to_name[id]

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
        return np.sqrt(
            np.nanmean((y_true - y_pred) ** 2)
        )

    @staticmethod
    def sort_recommendatdions(movie_names, similarity_values):
        recommendations = sorted(zip(movie_names, similarity_values), key=lambda x: x[1], reverse=True)
        recommendations = list(zip(*recommendations))
        return recommendations[0], recommendations[1]
