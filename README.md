# Recommender system with Collaborative Filtering

*ready to production*

---

### Dataset: MovieLens

---
### Learning methods available: 
* SVD
* ALS

---

## Docker:
    docker-compose up -d

---

## CLI:
Train model:

    python model.py train --dataset=<dataset path>

Evaluate model:

    python model.py evaluate --dataset=<dataset path>

Recommend movies for user:

    python model.py predict --user_id <user id> --top_m <top M movies to recommend>

Find similar movies to this one:

    python model.py find_similar --movie_name <movie name> --top_m <top M movies to recommend>

---

## API
Get system logs:

    GET /api/log 

Basic information (creator info, metrics, docker build datetime):

    GET /api/info
Warmup model:

    POST /api/reload

Get similar movies:

    POST /api/similar?movie=<movie name>&top_m=<top m similar movies>

Get recommendations for user:
    
    POST /api/predict?user_id=<user id on system DB>&top_m=<top m best recommendations>
    