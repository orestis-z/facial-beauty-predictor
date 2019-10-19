from queue import Queue
from threading import Thread
import joblib

from attractiveness_estimator.worker import worker_mtcnn_facenet_async_queue


def init_app(app):
    img_paths_queue = Queue()
    regressor_model = joblib.load(app.config["REGRESSOR_MODEL_PATH"])
    results_queue = worker_mtcnn_facenet_async_queue(
        img_paths_queue, app.config["FACENET_MODEL_PATH"], regressor_model)
    score_queues = app.config["SCORE_QUEUES"]
    thread = Thread(target=distribute_results,
                    args=(results_queue, score_queues,), name="DistributeResultsThread", daemon=True)
    thread.start()
    return img_paths_queue


def distribute_results(results_queue, score_queues):
    while True:
        result = results_queue.get()
        result_queue = score_queues[result["id"]]
        result_queue.put(result["score"])
