from application.config import DATASET_BASE_PATH
from application.recommendation_model import My_Rec_Model


class ModelService:
    def __init__(self, app):
        self.app = app
        self.model = My_Rec_Model(
            logger=self.app.logger
        )
        self.model.train(dataset_path=DATASET_BASE_PATH)

    def warmup_model(self):
        self.model.warmup()

    def get_similar_movies(self, movie_name, top_k):
        movie_id = self.model.movie_name_to_id(movie_name)
        movie_names, similarity_values = self.model.find_similar(movie_id, top_k)

        recommendations = sorted(zip(movie_names, similarity_values), key=lambda x: x[1], reverse=True)

        return list(map(
            lambda r: {
                'movie_name': r[0],
                'rating': r[1]
            },
            recommendations
        ))

    def get_recommendations(self, reviews, top_k):
        movie_ratings = reviews[1]
        movie_ids = list(map(
            self.model.movie_name_to_movie_id,
            reviews[0]
        ))

        recommendations = self.model.predict(
            (movie_ids, movie_ratings),
            top_k
        )
        recommendation_ratings = recommendations[1]
        recommendation_names = list(map(
            self.model.movie_id_to_movie_name,
            recommendations[0]
        ))

        return recommendation_names, recommendation_ratings
