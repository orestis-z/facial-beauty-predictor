import os
import logging
import time
from queue import Queue
from threading import Thread

import tensorflow as tf
from skimage.transform import resize
import numpy as np
import imageio

from attractiveness_estimator.mtcnn.mtcnn import MTCNN
import attractiveness_estimator.facenet as facenet
import attractiveness_estimator.img_utils as img_utils

MARGIN = 44
FACE_IMAGE_SIZE = 182
IMAGE_BATCH = 1000


def worker_mtcnn_facenet_async(data, facenet_model_path, skip_multiple_faces=False):
     # img_paths -> img_list -> croped_faces -> features -> avg_features
    img_paths_queue = Queue()
    img_list_queue = Queue()
    cropped_faces_queue = Queue()
    features_queue = Queue()
    avg_features_queue = Queue()

    image_loader_thread = Thread(
        target=image_loader_async, args=(img_paths_queue, img_list_queue))
    image_loader_thread.daemon = True
    image_loader_thread.start()

    face_detector_thread = Thread(
        target=mtcnn_face_detector_async, args=(img_list_queue, cropped_faces_queue), kwargs=dict(skip_multiple_faces=skip_multiple_faces))
    face_detector_thread.daemon = True
    face_detector_thread.start()

    export_features_thread = Thread(
        target=gen_facenet_features_async, args=(cropped_faces_queue, features_queue, facenet_model_path))
    export_features_thread.daemon = True
    export_features_thread.start()

    calc_avg_features_thread = Thread(
        target=calc_avg_features_async, args=(features_queue, avg_features_queue))
    calc_avg_features_thread.daemon = True
    calc_avg_features_thread.start()

    avg_features = []

    def collect_avg_features():
        while True:
            avg_features.append(avg_features_queue.get())
            avg_features_queue.task_done()
    collect_features_thread = Thread(target=collect_avg_features)
    collect_features_thread.daemon = True
    collect_features_thread.start()

    for i, d in enumerate(data):
        img_paths_queue.put(dict(id=i, paths=d["img_paths"]))

    img_paths_queue.join()
    img_list_queue.join()
    cropped_faces_queue.join()
    features_queue.join()
    avg_features_queue.join()

    return avg_features


def worker_mtcnn_async(data):
     # img_paths -> img_list -> features -> avg_features
    img_paths_queue = Queue()
    img_list_queue = Queue()
    features_queue = Queue()
    avg_features_queue = Queue()

    image_loader_thread = Thread(
        target=image_loader_async, args=(img_paths_queue, img_list_queue))
    image_loader_thread.daemon = True
    image_loader_thread.start()

    mtcnn_thread = Thread(
        target=mtcnn_features_async, args=(img_list_queue, features_queue))
    mtcnn_thread.daemon = True
    mtcnn_thread.start()

    calc_avg_features_thread = Thread(
        target=calc_avg_features_async, args=(features_queue, avg_features_queue))
    calc_avg_features_thread.daemon = True
    calc_avg_features_thread.start()

    avg_features = []

    def collect_avg_features():
        while True:
            avg_features.append(avg_features_queue.get())
            avg_features_queue.task_done()
    collect_features_thread = Thread(target=collect_avg_features)
    collect_features_thread.daemon = True
    collect_features_thread.start()

    for i, d in enumerate(data):
        img_paths_queue.put(dict(id=i, paths=d["img_paths"]))

    img_paths_queue.join()
    img_list_queue.join()
    features_queue.join()
    avg_features_queue.join()

    return avg_features


def worker_mtcnn_facenet_2_async(data, facenet_model_path):
     # img_paths -> img_list -> croped_faces -> features -> avg_features
    img_paths_queue = Queue()
    img_list_queue = Queue()
    cropped_faces_queue = Queue()
    features_queue = Queue()
    avg_features_queue = Queue()

    image_loader_thread = Thread(
        target=image_loader_async, args=(img_paths_queue, img_list_queue))
    image_loader_thread.daemon = True
    image_loader_thread.start()

    face_detector_thread = Thread(
        target=mtcnn_face_detector_async, args=(img_list_queue, cropped_faces_queue), kwargs=dict(skip_multiple_faces=True, store_features=True))
    face_detector_thread.daemon = True
    face_detector_thread.start()

    export_features_thread = Thread(
        target=gen_facenet_features_async, args=(cropped_faces_queue, features_queue, facenet_model_path))
    export_features_thread.daemon = True
    export_features_thread.start()

    calc_avg_features_thread = Thread(
        target=calc_avg_features_async, args=(features_queue, avg_features_queue))
    calc_avg_features_thread.daemon = True
    calc_avg_features_thread.start()

    avg_features = []

    def collect_avg_features():
        while True:
            avg_features.append(avg_features_queue.get())
            avg_features_queue.task_done()
    collect_features_thread = Thread(target=collect_avg_features)
    collect_features_thread.daemon = True
    collect_features_thread.start()
    for i, d in enumerate(data):
        img_paths_queue.put(dict(id=i, paths=d["img_paths"]))

    img_paths_queue.join()
    img_list_queue.join()
    cropped_faces_queue.join()
    features_queue.join()
    avg_features_queue.join()

    return avg_features


def image_loader_async(img_path_queue, img_list_queue):
    while True:
        img_path_dict = img_path_queue.get()
        img_paths = img_path_dict["paths"]
        profile_id = img_path_dict["id"]

        img_list = []
        for img_path in img_paths:
            if os.path.exists(img_path):
                img = imageio.imread(img_path)
                img_list.append(img)
            else:
                logging.warn("{} not found".format(img_path))
        img_list_queue.put(dict(id=profile_id, img_list=img_list))
        img_path_queue.task_done()


def mtcnn_face_detector_async(img_list_queue, cropped_faces_queue,
                              skip_multiple_faces=False, store_features=False):
    # based on https://github.com/cjekel/tindetheus/blob/master/tindetheus/facenet_clone/align/align_dataset_mtcnn.py and https://github.com/davidsandberg/facenet/blob/master/src/align/align_dataset_mtcnn.py
    mtcnn = MTCNN()

    while True:
        img_list_dict = img_list_queue.get()
        img_list = img_list_dict["img_list"]
        profile_id = img_list_dict["id"]

        faces = []
        features = [] if store_features else None

        for img in img_list:
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
            if store_features:
                features.append(fc1)

        assert len(img_list) >= len(faces)

        cropped_faces_queue.put(
            dict(id=profile_id, faces=faces, features=features))
        img_list_queue.task_done()


def mtcnn_features_async(img_list_queue, features_queue):
    mtcnn = MTCNN()

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
        with tf.Session() as sess:
            # Load the model
            facenet.load_model(model_path)

            # Get input and output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph(
            ).get_tensor_by_name("phase_train:0")

            while True:
                cropped_face_dict = cropped_faces_queue.get()
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


def calc_avg_features_async(features_queue, avg_features_queue):
    # a function to create a vector of n average features for each
    # tinder profile

    while True:
        features_dict = features_queue.get()
        features = features_dict["features"]  # (128, n_emb) for FaceNet
        profile_id = features_dict["id"]

        avg_features = np.mean(features, axis=0)

        avg_features_queue.put(
            dict(id=profile_id, avg_features=avg_features))
        features_queue.task_done()
