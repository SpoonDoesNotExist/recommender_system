FROM python:3.10

WORKDIR /app

COPY requirements.txt /app
RUN python -m pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

COPY . /app

EXPOSE 5000
ENTRYPOINT ["python3"]
CMD ["run.py"]