from application.model.recommendation_model import MyRecModel


class ModelService:
    def __init__(self):
        self.model = MyRecModel()
        # self.model.train(dataset_path=DATASET_BASE_PATH + 'data')

    def warmup_model(self):
        self.model.warmup()

    def get_similar_movies(self, movie_name, top_k):
        movie_id = self.model.movie_name_to_movie_id(movie_name)
        recommendations = self.model.find_similar(movie_id, top_k)

        return list(map(
            lambda r: {
                'movie_name': self.model.movie_id_to_movie_name(r[0]),
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
