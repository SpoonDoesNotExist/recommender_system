import logging

import numpy as np
import pandas as pd


class MyRecModel:

    def __init__(self):
        self.idx_to_movie_id = None
        self.movie_id_to_idx = None
        self.ratings = None
        self.users_similarities = None

    def predict(self, reviews, top_m):
        user_reviews = self._create_user_review(reviews)
        user_similarities = np.matmul(user_reviews, self.ratings.T)
        user_similarities = np.expand_dims(user_similarities, axis=0)
        user_scores = self._get_scores(self.ratings, user_similarities)[0]

        scores, indexes = self._get_top_k(user_scores, top_m)
        ids = self._get_movie_ids(indexes)
        return self.df_movie[self.df_movie.movie_id.isin(ids)]

    def evaluate(self, dataset_path):
        raise NotImplementedError('TODO evaluate()')

        logging.info("i'm logging evaluate results..............")

    def train(self, dataset_path):
        self._load_dataset(dataset_path)
        ratings = pd.pivot_table(
            data=self.df_rating, index='user_id',
            columns='movie_id', values='rating',
            fill_value=0
        )
        self.movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(ratings.columns)}
        self.idx_to_movie_id = {idx: movie_id for movie_id, idx in self.movie_id_to_idx.items()}

        self.ratings = self._normalize(ratings.to_numpy())

        self.users_similarities = self._get_similarities(self.ratings)

        logging.info("i'm logging train results..............")

    def warmup(self):
        raise NotImplementedError()

    def find_similar(self):
        raise NotImplementedError()

    def _create_user_review(self, reviews):
        movie_idxs = self._get_movie_indexes(reviews[0])
        movie_ratings = reviews[1]
        user_reviews = np.zeros(len(self.movie_id_to_idx))
        user_reviews[movie_idxs] = movie_ratings
        return self._normalize(user_reviews)

    def _load_dataset(self, dataset_path):
        rating_path = dataset_path + '/ratings.dat'
        self.df_rating = pd.read_csv(
            rating_path, sep='::', header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )
        self.df_rating.movie_id = self.df_rating.movie_id.apply(lambda x: f'uid-{x}')

        user_path = dataset_path + '/users.dat'
        self.df_user = pd.read_csv(
            user_path, sep='::', header=None,
            names=['user_id', 'gender', 'age', 'occupation', 'zipcode']
        )

        movie_path = dataset_path + '/movies.dat'
        self.df_movie = pd.read_csv(
            movie_path, sep='::', header=None,
            names=['movie_id', 'title', 'genre'],
            encoding='windows-1251'
        )
        self.df_movie.movie_id = self.df_movie.movie_id.apply(lambda x: f'uid-{x}')

    def _get_movie_indexes(self, movie_ids):
        return list(map(
            lambda x: self.movie_id_to_idx[x],
            movie_ids
        ))

    def _get_movie_ids(self, movie_indexes):
        return list(map(
            lambda x: self.idx_to_movie_id[x],
            movie_indexes
        ))

    @staticmethod
    def _get_top_k(array, k):
        idx = np.argpartition(array, -k)[-k:]
        return array[idx], idx

    @staticmethod
    def _normalize(matrix):
        return (matrix - matrix.mean(axis=-1, keepdims=True)) / matrix.max(axis=-1, keepdims=True)

    @staticmethod
    def _get_similarities(matrix):
        return np.matmul(matrix, matrix.T)

    @staticmethod
    def _get_scores(matrix, similarities):
        return np.matmul(similarities, matrix) / similarities.sum(axis=-1, keepdims=True)
