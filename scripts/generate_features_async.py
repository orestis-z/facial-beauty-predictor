import os
import argparse
import pickle

import numpy as np
from scipy.special import logit

from attractiveness_estimator.worker import worker_mtcnn_async, worker_mtcnn_facenet_async, worker_mtcnn_facenet_2_async


def main(args):
    db_dict = pickle.load(open(args.db_file, "rb"))
    db = db_dict["all"]

    if args.backbone == "mtcnn-facenet":
        features_dict = worker_mtcnn_facenet_async(
            db, args.facenet_model_path, skip_multiple_faces=bool(args.skip_multiple_faces))
    elif args.backbone == "mtcnn":
        features_dict = worker_mtcnn_async(db)
    elif args.backbone == "mtcnn-facenet-2":
        features_dict = worker_mtcnn_facenet_2_async(
            db, args.facenet_model_path)
    else:
        raise ValueError

    output_dir = os.path.join(args.output_dir, args.backbone)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    features_path = os.path.join(output_dir, "features.npy")
    print("Saving features to {}".format(features_path))
    np.save(open(features_path, "wb"), features_dict)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--db',
        dest='db_file',
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        default='data',
        type=str
    )
    parser.add_argument(
        '--skip-multiple-faces',
        dest='skip_multiple_faces',
        action="store_true",
    )
    parser.add_argument(
        '--facenet-model-path',
        dest='facenet_model_path',
        default="data/20170512-110547",
        type=str
    )
    parser.add_argument(
        '--backbone',
        help="One of {mtcnn, mtcnn-facenet, mtcnn-facenet-2}",
        dest='backbone',
        default="mtcnn-facenet",
        type=str
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    main(args)
