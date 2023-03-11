from logging.handlers import RotatingFileHandler

from flask import Flask
import logging

from application.config import LOGGING_FILE_PATH

app = Flask(__name__)

logfile = open(LOGGING_FILE_PATH, 'w')
logfile.close()
handler = RotatingFileHandler(LOGGING_FILE_PATH)
formatter = logging.Formatter(
    fmt='>>> %(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)

default_logger = logging.getLogger()
default_logger.setLevel(logging.INFO)

app.logger.addHandler(handler)

import application.controller
