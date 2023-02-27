from logging.handlers import RotatingFileHandler

from flask import Flask
import logging

from config import LOGGING_FILE_PATH

app = Flask(__name__)

handler = RotatingFileHandler(LOGGING_FILE_PATH)
formatter = logging.Formatter(
    fmt='>>> %(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)

default_logger = logging.getLogger()
default_logger.setLevel(logging.INFO)

app.logger.addHandler(handler)

import controller
