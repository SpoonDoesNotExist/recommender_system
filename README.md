# Recommender system with Collaborative Filtering

*ready to production*

---

### Dataset: MovieLens

---
#### Learning methods available: 
* SVD
* ALS

---

### Docker:
    docker-compose up -d

---

### CLI:
Train model:

    python model.py train --dataset=<dataset path>

Evaluate model:

    python model.py evaluate --dataset=<dataset path>

Recommend movies for user:

    python model.py predict --user_id <user id> --top_m <top M movies to recommend>

Find similar movies to this one:

    python model.py find_similar --movie_name <movie name> --top_m <top M movies to recommend>
