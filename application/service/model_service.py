from application.config import DATASET_BASE_PATH, TRAIN_METRICS_PATH, TEST_METRICS_PATH
from application.recommendation_model import My_Rec_Model


class ModelService:
    def __init__(self, app):
        self.app = app
        self.model = My_Rec_Model(
            logger=self.app.logger
        )

        try:
            self.model.warmup()
        except:
            kwargs_svd = {
                'method_name': 'svd',
                'n_latent_factors': 40
            }
            train_eval_metric = self.model.train(dataset_path=DATASET_BASE_PATH, **kwargs_svd)

            test_path = DATASET_BASE_PATH + '/ratings_test.dat'
            test_eval_metric = self.model.evaluate(dataset_path=test_path)
            app.logger.info(f'Test RMSE: {test_eval_metric}\nDataset: {test_path}')

            self._save_metrics(train_eval_metric, test_eval_metric)

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

    def get_recommendations(self, user_id, top_k):
        movie_names, similarity_values = self.model.predict(user_id, top_k)
        return movie_names, similarity_values

    def _save_metrics(self, train_metrics, test_metrics):
        with open(TRAIN_METRICS_PATH, 'w') as f:
            f.write(str(train_metrics))
        with open(TEST_METRICS_PATH, 'w') as f:
            f.write(str(test_metrics))
