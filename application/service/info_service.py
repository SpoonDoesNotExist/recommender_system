class InfoService:
    def __init__(self, app):
        self.app = app

    def get_credentials(self):
        return {
            'Name': 'Eduard',
            'Surname': 'Khusnutdinov',
            'University':{
                'Name': 'Tomsk State University',
                'Grade': '3',
            },
        }
