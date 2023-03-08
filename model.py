import typer as typer

from application.config import DATASET_BASE_PATH
from application.recommendation_model import My_Rec_Model

recsys_model = My_Rec_Model()
recsys_model.warmup()

cli_app = typer.Typer()


@cli_app.command()
def warmup():
    recsys_model.warmup()


@cli_app.command()
def train(dataset=DATASET_BASE_PATH):
    recsys_model.train(dataset)


@cli_app.command()
def evaluate(dataset=DATASET_BASE_PATH):
    recsys_model.evaluate(dataset)


@cli_app.command()
def find_similar(movie_id, N):
    movie_names, similarity_values = recsys_model.find_similar(movie_id, N)
    return movie_names, similarity_values


@cli_app.command()
def predict(user_id, top_m):
    top_m = int(top_m)
    movie_names, similarity_values = recsys_model.predict(user_id, top_m)
    print(movie_names, similarity_values)
    return movie_names, similarity_values


if __name__ == '__main__':
    cli_app()
