from flask import jsonify

from app import app
from service.info_service import InfoService

info_service = InfoService()


@app.route('/api/info', methods=['GET'])
def info():
    return jsonify(info_service.get_credentials())
