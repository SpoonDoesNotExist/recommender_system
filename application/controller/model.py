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
    top_m = int(args.get('top_m'))

    top_similar_movies = model_service.get_similar_movies(movie_name, top_m)

    return jsonify(top_similar_movies)


@app.route('/api/predict', methods=['POST'])
def predict():
    args = request.args
    body = request.get_json()

    top_m = args.get('top_m')
    reviews = body['reviews']

    recommendations = model_service.get_recommendations(reviews, top_m)

    return jsonify(recommendations)
