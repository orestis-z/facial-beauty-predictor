import os
import logging
import time
from queue import Queue
from threading import Thread
import logging
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import time

import tensorflow as tf
from skimage.transform import resize
import numpy as np
import imageio

from attractiveness_estimator.backbone.mtcnn.mtcnn import MTCNN
import attractiveness_estimator.backbone.facenet as facenet
import attractiveness_estimator.utils.img as img_utils

MARGIN = 44
FACE_IMAGE_SIZE = 182
IMAGE_BATCH = 1000


def worker_mtcnn_facenet_async(db, facenet_model_path, skip_multiple_faces=False):
    # img_paths -> img_list -> croped_faces -> features -> avg_features
    img_paths_queue = Queue()
    pipeline = PipelineJoinable(img_paths_queue,
                                image_loader_async,
                                (mtcnn_face_detector_async, dict(kwargs=dict(
                                    skip_multiple_faces=skip_multiple_faces))),
                                (gen_facenet_features_async,
                                    dict(args=[facenet_model_path])),
                                calc_avg_features_async)
    for profile_id, profile in db.items():
        img_paths_queue.put(dict(id=profile_id, paths=profile["img_paths"]))
    pipeline.join()
    features_list = pipeline.get_output()
    features_dict = {features["id"]: features["features"]
                     for features in features_list}

    return features_dict


def worker_mtcnn_facenet_async_queue(img_paths_queue, facenet_model_path, regressor_model, percentiles, skip_multiple_faces=True):
    # img_paths -> img_list -> croped_faces -> features -> avg_features
    pipeline = Pipeline(img_paths_queue,
                        (image_loader_async, dict(kwargs=dict(local_files=False))),
                        (mtcnn_face_detector_async, dict(kwargs=dict(
                            skip_multiple_faces=skip_multiple_faces))),
                        (gen_facenet_features_async,
                         dict(args=[facenet_model_path])),
                        calc_avg_features_async,
                        (regress_score_async, dict(args=[regressor_model])),
                        (calc_percentile, dict(args=[percentiles])))
    return pipeline.out_queue


def worker_mtcnn_async(db):
    # img_paths -> img_list -> features -> avg_features
    img_paths_queue = Queue()
    pipeline = PipelineJoinable(img_paths_queue,
                                image_loader_async,
                                mtcnn_features_async,
                                calc_avg_features_async)
    for profile_id, profile in db.items():
        img_paths_queue.put(dict(id=profile_id, paths=profile["img_paths"]))
    pipeline.join()
    features_list = pipeline.get_output()
    features_dict = {features["id"]: features["features"]
                     for features in features_list}

    return features_dict


def worker_mtcnn_facenet_2_async(db, facenet_model_path):
    # img_paths -> img_list -> croped_faces -> features -> avg_features
    img_paths_queue = Queue()
    pipeline = PipelineJoinable(img_paths_queue,
                                image_loader_async,
                                (mtcnn_face_detector_async, dict(kwargs=dict(
                                    skip_multiple_faces=True, store_features=True))),
                                (gen_facenet_features_async,
                                    dict(args=[facenet_model_path])),
                                calc_avg_features_async)
    for profile_id, profile in db.items():
        img_paths_queue.put(dict(id=profile_id, paths=profile["img_paths"]))
    pipeline.join()
    features_list = pipeline.get_output()
    features_dict = {features["id"]: features["features"]
                     for features in features_list}

    return features_dict


class Pipeline():
    def __init__(self, in_queue, *parts):
        self.queue_list = [in_queue]
        self.out_queue = in_queue
        for i, part in enumerate(parts):
            if hasattr(part, "__len__"):
                func = part[0]
                args = part[1].get("args", [])
                kwargs = part[1].get("kwargs", {})
            else:
                func = part
                args = []
                kwargs = {}
            in_queue = self.out_queue
            self.out_queue = Queue()
            self.queue_list.append(self.out_queue)
            thread = Thread(
                target=func, args=(in_queue, self.out_queue, *args),
                kwargs=kwargs, name="Pipeline{}Thread".format(i), daemon=True)
            thread.start()


class PipelineJoinable(Pipeline):
    def __init__(self, in_queue, *parts):
        super.__init__(in_queue, *parts)
        self.out_list = []
        thread = Thread(
            target=collect_async, args=(self.out_queue, self.out_list), name="PipelineCollectThread", daemon=True)
        thread.start()

    def join(self):
        for queue in self.queue_list:
            queue.join()

    def get_output(self):
        return self.out_list


def image_loader_async(img_path_queue, img_list_queue, local_files=True):
    while True:
        img_path_dict = img_path_queue.get()

        def load_images(img_paths, profile_id):
            start = time.time()
            img_list = []

            def load_image(img_path):
                if local_files:
                    if os.path.exists(img_path):
                        img = imageio.imread(img_path)
                        img_list.append(img)
                    else:
                        logging.warn("{} not found".format(img_path))
                else:
                    try:
                        response = requests.get(img_path, timeout=5)
                        img = imageio.imread(BytesIO(response.content))
                        img_list.append(img)
                    except Exception:
                        logging.exception(
                            "Could not download {}".format(img_path))

            with ThreadPoolExecutor(thread_name_prefix="LoadImagesWorker", max_workers=max(len(img_paths), os.cpu_count() + 4)) as executor:
                executor.map(load_image, img_paths)

            img_list_queue.put(dict(id=profile_id, img_list=img_list))
            img_path_queue.task_done()
            logging.debug("Loading {} profile images took {:.3f}s".format(
                len(img_list),
                time.time() - start))
        thread = Thread(target=load_images, name="LoadImagesThread", args=(
            img_path_dict["paths"], img_path_dict["id"]))
        thread.start()


def mtcnn_face_detector_async(img_list_queue, cropped_faces_queue,
                              skip_multiple_faces=False, store_features=False):
    # based on https://github.com/cjekel/tindetheus/blob/master/tindetheus/facenet_clone/align/align_dataset_mtcnn.py and https://github.com/davidsandberg/facenet/blob/master/src/align/align_dataset_mtcnn.py
    mtcnn = MTCNN()
    logging.debug("MTCNN initialization done")

    while True:
        img_list_dict = img_list_queue.get()
        start = time.time()
        img_list = img_list_dict["img_list"]
        profile_id = img_list_dict["id"]

        faces = []
        features = [] if store_features else None

        for i, img in enumerate(img_list):
            img = img[:, :, 0:3]
            bounding_boxes, _, fc1 = mtcnn.detect_faces_raw(img)
            n_faces = bounding_boxes.shape[0]
            if n_faces > 0:
                det = bounding_boxes[:, 0:4]
                img_size = np.asarray(img.shape)[0:2]
                if n_faces > 1:
                    if skip_multiple_faces:
                        continue
                    bounding_box_size = (
                        det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                    img_center = img_size / 2
                    offsets = np.vstack(
                        [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                    offset_dist_squared = np.sum(
                        np.power(offsets, 2.0), 0)
                    # some extra weight on the centering
                    index = np.argmax(
                        bounding_box_size - offset_dist_squared * 2.0)
                    det = det[index, :]
                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - MARGIN / 2, 0)
                bb[1] = np.maximum(det[1] - MARGIN / 2, 0)
                bb[2] = np.minimum(det[2] + MARGIN / 2, img_size[1])
                bb[3] = np.minimum(det[3] + MARGIN / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                scaled = resize(cropped, (FACE_IMAGE_SIZE, FACE_IMAGE_SIZE),
                                mode='constant')
                faces.append(scaled)
            else:
                logging.debug(
                    "No faces found in image nr {} of {}".format(i + 1, profile_id))
            if store_features:
                features.append(fc1)

        assert len(img_list) >= len(faces)

        cropped_faces_queue.put(
            dict(id=profile_id, faces=faces, features=features))
        img_list_queue.task_done()
        logging.debug("Detecting {} faces in {} images took {:.3f}s".format(
            len(faces), len(img_list), time.time() - start))


def mtcnn_features_async(img_list_queue, features_queue):
    mtcnn = MTCNN()
    logging.debug("MTCNN initialization done")

    while True:
        img_list_dict = img_list_queue.get()
        img_list = img_list_dict["img_list"]
        profile_id = img_list_dict["id"]

        features = []

        for img in img_list:
            img = img[:, :, 0:3]
            bounding_boxes, _, fc1 = mtcnn.detect_faces_raw(img)
            n_faces = bounding_boxes.shape[0]
            if n_faces == 1:
                features.append(fc1)

        assert len(img_list) >= len(features)

        features_queue.put(dict(id=profile_id, features=features))
        img_list_queue.task_done()


def gen_facenet_features_async(cropped_faces_queue, features_queue,
                               model_path, image_size=160):
    # based on https://github.com/cjekel/tindetheus/blob/master/tindetheus/export_features.py and https://github.com/davidsandberg/facenet/blob/master/src/compare.py
    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            # Load the model
            facenet.load_model(model_path)

            logging.debug("FaceNet initialization done")

            # Get input and output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph(
            ).get_tensor_by_name("phase_train:0")

            while True:
                cropped_face_dict = cropped_faces_queue.get()
                start = time.time()
                img_list = cropped_face_dict["faces"]
                mtcnn_features = cropped_face_dict.get("features")
                profile_id = cropped_face_dict["id"]

                # Run forward pass to calculate embeddings
                n_images = len(img_list)

                # print('Number of images: ', n_images)
                batch_size = IMAGE_BATCH
                if n_images % batch_size == 0:
                    n_batches = n_images // batch_size
                else:
                    n_batches = (n_images // batch_size) + 1
                # print('Number of batches: ', n_batches)
                embedding_size = embeddings.get_shape()[1]
                emb_array = np.zeros((n_images, embedding_size))
                # start_time = time.time()

                for i in range(n_batches):
                    if i == n_batches - 1:
                        n = n_images
                    else:
                        n = i * batch_size + batch_size
                    # Get images for the batch
                    imgs_batch = img_list[i * batch_size:n]
                    n_samples = len(imgs_batch)
                    images = np.zeros((n_samples, image_size, image_size, 3))
                    for j in range(n_samples):
                        img = imgs_batch[j]
                        if img.ndim == 2:
                            img = img_utils.to_rgb(img)
                        img = img_utils.prewhiten(img)
                        img = img_utils.crop(img, False, image_size)
                        img = img_utils.flip(img, False)
                        images[j, :, :, :] = img
                    feed_dict = {images_placeholder: images,
                                 phase_train_placeholder: False}
                    # Use the facenet model to calculate embeddings
                    embed = sess.run(embeddings, feed_dict=feed_dict)
                    emb_array[i * batch_size:n, :] = embed
                    # print('Completed batch', i+1, 'of', n_batches)

                if mtcnn_features is not None:
                    if n_images:
                        features = np.hstack((mtcnn_features, emb_array))
                    else:
                        features = np.empty((0, emb_array.shape[1] + 256))
                else:
                    features = emb_array

                # run_time = time.time() - start_time
                # print('Run time: ', run_time)

                features_queue.put(dict(id=profile_id, features=features))
                cropped_faces_queue.task_done()
                logging.debug("Feature vector generation for {} faces took {:.3f}s".format(
                    n_images, time.time() - start))


def calc_avg_features_async(features_queue, avg_features_queue):
    # a function to create a vector of n average features for each
    # tinder profile

    while True:
        features_dict = features_queue.get()
        features = features_dict["features"]  # (128, n_emb) for FaceNet
        profile_id = features_dict["id"]

        if features.shape[0]:
            avg_features = np.mean(features, axis=0)
        else:
            avg_features = features

        avg_features_queue.put(
            dict(id=profile_id, features=avg_features))
        features_queue.task_done()


def regress_score_async(features_queue, score_queue, model):
    while True:
        features_dict = features_queue.get()
        start = time.time()
        features = features_dict["features"]  # (1, 128) for FaceNet
        profile_id = features_dict["id"]

        if features.shape[0]:
            try:
                features = features.reshape(1, -1)
                score_pred = model.predict(features)
                score_pred = np.clip(score_pred, 0, 1)
            except Exception:
                logging.exception("Failed to predict profile score")
                score_pred = None
        else:
            score_pred = 0

        score_queue.put(
            dict(id=profile_id, score=score_pred))
        features_queue.task_done()
        logging.debug("Regression took {:.3f}s".format(time.time() - start))


def calc_percentile(score_queue, percentile_queue, percentiles):
    while True:
        score_dict = score_queue.get()
        profile_id = score_dict["id"]
        score = score_dict["score"]

        if score is None:
            percentile = None
        else:
            percentile = 1
            for i, percentile_score in enumerate(percentiles):
                if score > percentile_score:
                    percentile = 1 - i * 0.05
                else:
                    break
        percentile_queue.put(
            dict(id=profile_id, percentile=percentile))
        score_queue.task_done()


def collect_async(in_queue, out_list):
    while True:
        out_list.append(in_queue.get())
        in_queue.task_done()
        in_queue.task_done()
