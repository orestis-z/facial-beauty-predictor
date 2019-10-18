import argparse
import os
import importlib
import pickle
import joblib
import math

from scipy.special import expit
import sklearn.metrics as metrics
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt


def main(args):
    model_files = sorted([model_file for model_file in os.listdir(
        args.models_dir) if model_file[0] != "." and os.path.isfile(os.path.join(args.models_dir, model_file))])

    db_dict = pickle.load(open(args.db_file, "rb"))
    db = db_dict["test"]
    features = np.load(open(args.features, "rb"), allow_pickle=True).item()
    features = [features[profile_id] for profile_id in db.keys()]
    nan_idx = np.argwhere(np.isnan(features))
    nan_idx = np.unique(nan_idx[:, 0])
    print("Found {} images without faces in test split".format(len(nan_idx)))
    features = np.delete(features, nan_idx, axis=0)
    db_flat = np.delete(list(db.values()), nan_idx)

    scores = np.array([d["score"] for d in db_flat])

    sorted_inds = np.argsort(scores)
    scores_sorted = scores[sorted_inds]

    x = np.arange(0, 1, 1 / len(features))

    n_plots = len(model_files)
    n_cols = 3  # min(math.ceil(np.sqrt(n_plots)), 5)
    n_rows = max(math.ceil(n_plots / n_cols), 2)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(14, 8))
    fig.suptitle('Model comparison')

    FONTSIZE = 8
    plt.rcParams.update({'font.size': FONTSIZE})

    for i, model_file in enumerate(model_files):
        row = int(i / n_cols)
        col = i - row * n_cols
        ax = axs[row, col]

        model_name = model_file[:-6].split(".")[-1]
        order = int(model_file[-5])

        model = joblib.load(os.path.join(args.models_dir, model_file))
        logits_pred = model.predict(features)
        scores_pred = np.array([expit(y) for y in logits_pred])
        rmse = np.sqrt(metrics.mean_squared_error(scores_pred, scores))
        mae = metrics.mean_absolute_error(scores_pred, scores)
        pc = pearsonr(scores_pred, scores)

        ax.plot(scores_sorted,
                scores_pred[sorted_inds], ".", label="pred", markersize=2)
        ax.plot(scores_sorted, scores_sorted, label="GT")
        ax.legend()
        ax.set_ylabel('score', fontsize=FONTSIZE)
        ax.set_xlabel('score GT', fontsize=FONTSIZE)
        ax.set_title("{} ({})\nRMSE: {:.4f}, MAE: {:.4f}, PC {:.3f}".format(
            model_name, order, rmse, mae, pc[0]))
        ax.grid(linestyle='--')
        for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(FONTSIZE)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(args.output_dir, "model_comparison.png"))
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--db',
        dest='db_file',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        default='data',
        type=str
    )
    parser.add_argument(
        '--models-dir',
        dest='models_dir',
        default='data/models',
        type=str
    )
    parser.add_argument(
        '--features',
        dest='features',
        default='data/features.npy',
        type=str
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    main(args)
