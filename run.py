from application.app import app
from waitress import serve

run_mode = 'prod'

if __name__ == '__main__':
    if run_mode == 'dev':
        app.run(host='0.0.0.0', port=5000)
    elif run_mode == 'prod':
        serve(app, host='0.0.0.0', port=5000)
    else:
        raise NotImplementedError(f'Unexpected run mode: {run_mode}')
