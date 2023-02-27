from flask import make_response

from app import app
from config import RETURN_LOG_ROWS
from service.log_service import LogService

log_service = LogService()


@app.route('/api/log', methods=['GET'])
def log():
    logs = log_service.get_logs(RETURN_LOG_ROWS)

    response = make_response(logs, 200)
    response.mimetype = "text/plain"
    return response
