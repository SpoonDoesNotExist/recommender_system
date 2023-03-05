class InfoService:
    def __init__(self, app):
        self.app = app

    def get_credentials(self):
        return {
            'credentials': 'c',
            'docker_image_datetime': 'datetime',
            'metrics': {
                'datetime': 'd',
                'results': 'r'
            }
        }
