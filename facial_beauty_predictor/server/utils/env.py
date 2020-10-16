import os


def is_main_run():
    return os.environ.get("FLASK_RUN_FROM_CLI") != "true" or \
        os.environ.get("WERKZEUG_RUN_MAIN") == "true"


def is_reloader():
    return os.environ.get("FLASK_ENV") == "development" and \
        os.environ.get("WERKZEUG_RUN_MAIN") != "true" and \
        os.environ.get("WERKZEUG_SERVER_FD") is not None
