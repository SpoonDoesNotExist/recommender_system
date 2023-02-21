from flask import Flask, request, make_response
import logging

from config import LOGGING_FILE_PATH, RETURN_LOG_ROWS

app = Flask(__name__)


@app.route('/api/log', methods=['GET'])
def log():
    with open(LOGGING_FILE_PATH, 'r') as file:
        return_rows = file.readlines()[-RETURN_LOG_ROWS:]
        return_data = '\n'.join(return_rows)

    response = make_response(return_data, 200)
    response.mimetype = "text/plain"
    return response


@app.route('/api/info', methods=['GET'])
def info():
    raise NotImplementedError()


@app.route('/api/reload', methods=['POST'])
def reload():
    raise NotImplementedError()


@app.route('/api/predict', methods=['POST'])
def predict():
    args = request.args

    top_m = args.get('top_m')
    body = request.get_json(force=True)
    print(top_m, body)
    raise NotImplementedError()


@app.route('/api/similar', methods=['POST'])
def similar():
    body = request.get_json(force=True)
    print(body)
    raise NotImplementedError()


@app.errorhandler(500)
def internal_server_error(e):
    return f'[error 500] {str(e)}', 500


if __name__ == '__main__':
    logging.basicConfig(filename=LOGGING_FILE_PATH, encoding='utf-8', level=logging.BASIC_FORMAT)
    app.run()
