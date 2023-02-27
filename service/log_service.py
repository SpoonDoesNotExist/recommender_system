from config import LOGGING_FILE_PATH


class LogService:
    def __init__(self):
        self.log_file_path = LOGGING_FILE_PATH

    def get_logs(self, num_rows):
        with open(self.log_file_path, 'r') as file:
            log_rows = file.readlines()[-num_rows:]
            log_rows = ''.join(log_rows)
            return log_rows
