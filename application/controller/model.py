from flask import request, jsonify

from application.app import app
from application.service.model_service import ModelService

model_service = ModelService(app)


@app.route('/api/reload', methods=['POST'])
def reload():
    model_service.warmup_model()
    return jsonify(success=True)


@app.route('/api/similar', methods=['POST'])
def similar():
    args = request.args

    movie_name = args.get('movie')
    top_m = args.get('top_m', type=int)

    top_similar_movies = model_service.get_similar_movies(movie_name, top_m)

    return jsonify(top_similar_movies)


@app.route('/api/predict', methods=['POST'])
def predict():
    args = request.args

    top_m = args.get('top_m', type=int)
    user_id = args.get('user_id')

    recommendations = model_service.get_recommendations(user_id, top_m)

    return jsonify(recommendations)
