import os
import time
import logging
from requests.exceptions import RequestException
from queue import SimpleQueue
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
import time

from flask import Flask, request
from flask_json import FlaskJSON, json_response, JsonError
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
import numpy as np

from attractiveness_estimator.server.init_app import fetch_files, init_worker
from attractiveness_estimator.server.utils.env import is_main_run


DEV = os.environ.get("FLASK_ENV") == "development"
logging.basicConfig(level=logging.DEBUG if DEV else logging.INFO)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger("s3transfer").setLevel(logging.INFO)
logging.getLogger("botocore").setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.INFO)
# logging.getLogger("werkzeug").setLevel(logging.INFO)
# logging.getLogger("metric").setLevel(logging.INFO)


def create_app():
    start = time.time()

    main_run = is_main_run()

    # create and configure the app
    app = Flask(__name__)
    app.config.from_object(os.environ.get(
        'APP_SETTINGS', "attractiveness_estimator.server.config.Config"))

    if main_run:
        logging.info("Creating app with config:\n" +
                     '\n'.join("    {}: {}".format(k, v)
                               for k, v in app.config.items()))

    if main_run:
        app.errorhandler(Exception)(_on_exception)

        # if app.config["LOG_METRICS"]:
        #     metrics.init_app(app)
        #     request_check.init_app(app)

        fetch_files(app)
        app.config["PERCENTILE_QUEUES"] = {}
        img_paths_queue = init_worker(app)
        app.config["IMG_PATHS_QUEUE"] = img_paths_queue

        json = FlaskJSON()
        json.init_app(app)

        init_app(app)

        if not app.config["DEBUG"]:
            sentry_sdk.init(
                dsn="https://af0dce792f934abaac5bb9d7bcad7df8@sentry.io/1785265",
                integrations=[FlaskIntegration()],
            )

        logging.info("App initialization done. Took {0:.1f}s".format(
            time.time() - start))

    return app


def _on_exception(e):
    logging.exception(type(e).__name__)
    return json_response(500)


def init_app(app):
    @app.route('/', methods=["POST"])
    def estimate_score():
        start = time.time()
        paths = request.json.get('paths')
        task_id = uuid.uuid4()

        assert len(paths)

        result_queue = SimpleQueue()
        app.config["PERCENTILE_QUEUES"][task_id] = result_queue
        img_paths_queue = app.config["IMG_PATHS_QUEUE"]
        img_paths_queue.put(dict(paths=paths, id=task_id))
        percentile = result_queue.get()
        if percentile is None:
            raise JsonError(500)
        data = dict(percentile=percentile)
        logging.info("Percentile: {:.0f}. Estimation took {:.3f}s".format(
            percentile * 100, time.time() - start))
        return json_response(data=data)

    @app.route('/ping', methods=["GET"])
    def ping():
        return "", 200
