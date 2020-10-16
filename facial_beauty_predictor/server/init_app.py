import os
from queue import Queue
from threading import Thread
import joblib
from urllib.parse import urlparse
import urllib.request
import logging

import boto3
import numpy as np

from facial_beauty_predictor.worker import worker_mtcnn_facenet_async_queue


def fetch_files(app):
    if is_remote(app.config["FACENET_MODEL_PATH"]):
        local_facenet_model_path = "data/facenet"
        if not folder_exists(local_facenet_model_path):
            download_s3_folder(
                app.config["FACENET_MODEL_PATH"], local_facenet_model_path)
            logging.debug("Downloaded facenet model to {}".format(
                local_facenet_model_path))
        app.config["FACENET_MODEL_PATH"] = local_facenet_model_path
    if is_remote(app.config["REGRESSOR_MODEL_PATH"]):
        local_regressor_model_path = "data/regressor_model.pkl"
        if not file_exists(local_regressor_model_path):
            download_s3_file(
                app.config["REGRESSOR_MODEL_PATH"], local_regressor_model_path)
            logging.debug("Downloaded regressor model to {}".format(
                local_regressor_model_path))
        app.config["REGRESSOR_MODEL_PATH"] = local_regressor_model_path
    if is_remote(app.config["PERCENTILS_PATH"]):
        local_percentiles_path = "data/percentiles.npy"
        if not file_exists(local_percentiles_path):
            download_s3_file(
                app.config["PERCENTILS_PATH"], local_percentiles_path)
            logging.debug("Downloaded percentiles to {}".format(
                local_percentiles_path))
        app.config["PERCENTILS_PATH"] = local_percentiles_path


def init_worker(app):
    img_paths_queue = Queue()
    regressor_model = joblib.load(app.config["REGRESSOR_MODEL_PATH"])
    percentiles = np.load(open(app.config["PERCENTILS_PATH"], "rb"))
    results_queue = worker_mtcnn_facenet_async_queue(
        img_paths_queue, app.config["FACENET_MODEL_PATH"], regressor_model, percentiles)
    result_queues = app.config["PERCENTILE_QUEUES"]
    thread = Thread(target=distribute_results,
                    args=(results_queue, result_queues,), name="DistributeResultsThread", daemon=True)
    thread.start()
    return img_paths_queue


def distribute_results(results_queue, result_queues):
    while True:
        result = results_queue.get()
        result_queue = result_queues[result["id"]]
        result_queue.put(result["percentile"])


def is_remote(url):
    return bool(urlparse(url).netloc)


def file_exists(path):
    return os.path.exists(path) and os.path.isfile(path)


def folder_exists(path):
    return os.path.exists(path) and os.path.isdir(path)


def download_s3_folder(path, dest):
    path = path[5:]
    bucket_name = path.split("/")[0]
    s3_resource = boto3.resource("s3")
    bucket = s3_resource.Bucket(bucket_name)
    remote_directory_name = "/".join(path.split("/")[1:])
    for obj in bucket.objects.filter(Prefix=remote_directory_name):
        file_name = obj.key[len(remote_directory_name) + 1:]
        dest_folder = os.path.join(
            dest, os.path.dirname(file_name))
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        bucket.download_file(obj.key, os.path.join(
            dest, file_name))


def download_s3_file(path, dest):
    path = path[5:]
    bucket_name = path.split("/")[0]
    s3_resource = boto3.resource("s3")
    bucket = s3_resource.Bucket(bucket_name)
    remote_file_name = "/".join(path.split("/")[1:])
    bucket.download_file(remote_file_name, dest)
