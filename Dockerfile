FROM python:3.10

WORKDIR /app

COPY requirements.txt /app
RUN python -m pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

COPY . /app

RUN mkdir -p /app/application/data
RUN mkdir -p /app/application/model

EXPOSE 5000
CMD ["python3", "run.py"]