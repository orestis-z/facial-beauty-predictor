import os
import argparse
import pickle

import numpy as np
import imageio
import joblib
from scipy.special import expit
import matplotlib.pyplot as plt


def main(args):
    db_dict = pickle.load(open(args.db_file, "rb"))
    db = db_dict["all"]

    features = np.load(open(args.features, "rb"), allow_pickle=True).item()
    features = [features[profile_id] for profile_id in db.keys()]
    nan_idx = np.argwhere(np.isnan(features))
    nan_idx = np.unique(nan_idx[:, 0])
    print("Found {}/{} images without faces".format(len(nan_idx), len(db)))
    features = np.delete(features, nan_idx, axis=0)
    db_flat = np.delete(list(db.values()), nan_idx)

    model = joblib.load(args.model_file)
    logits_pred = model.predict(features)
    scores_pred = np.array([expit(y) for y in logits_pred]) * 10

    print("Score mean: {:.1f}".format(np.mean(scores_pred)))
    print("Score median: {:.1f}".format(np.median(scores_pred)))
    print("Score standard deviation: {:.1f}".format(np.std(scores_pred)))

    sorted_scores = sorted(scores_pred)
    plt.plot(np.arange(0, 1, 1 / len(sorted_scores)), sorted_scores)
    plt.grid(linestyle="--")
    plt.xlabel("prob to have a score between 0 and y")
    plt.ylabel("score")
    plt.show()

    # visualize some random profiles
    for i in range(len(db_flat)):
        random_idx = np.random.choice(range(len(db_flat)))
        # random_idx = i
        img_paths = db_flat[random_idx]["img_paths"]
        score = scores_pred[random_idx]
        # if score > 4:
        #     continue

        img_paths = [
            img_path for img_path in img_paths if os.path.exists(img_path)]
        if not len(img_paths):
            continue

        img_list = []
        for img_path in img_paths:
            img = imageio.imread(img_path)
            img_list.append(img)
        plt.figure(figsize=(14, 8))
        fig = plt.imshow(np.vstack(img_list))
        plt.title("score {:.1f}".format(score))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        plt.show()


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
        default='data/tinder',
        type=str
    )
    parser.add_argument(
        '--model',
        dest='model_file',
        type=str
    )
    parser.add_argument(
        '--features',
        dest='features',
        type=str
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
