import os
import sys
import pickle
import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logit

from utils import normalize_dataset


def main(args):
    db_dict = pickle.load(open(args.db_file, "rb"))
    db_dict = normalize_dataset(db_dict)["data"]

    sorted_scores = sorted([d["score"] for d in db_dict["all"].values()])
    x = np.linspace(0, 1, len(sorted_scores)) * 100

    _, ax = plt.subplots(figsize=(14, 8))
    ax.plot(x, sorted_scores, label=args.db_file)

    mean_scores = sorted_scores
    x_mean = x

    if args.db_extra_file:
        db_extra_dict = pickle.load(open(args.db_extra_file, "rb"))
        db_extra_dict = normalize_dataset(db_extra_dict)["data"]
        sorted_scores_extra = sorted([d["score"]
                                      for d in db_extra_dict["all"].values()])
        x_extra = np.linspace(0, 1, len(sorted_scores_extra)) * 100
        ax.plot(x_extra, sorted_scores_extra, label=args.db_extra_file)

        if len(sorted_scores_extra) > len(sorted_scores):
            sorted_scores_extra = np.interp(x, x_extra, sorted_scores_extra)
        else:
            sorted_scores = np.interp(x_extra, x, sorted_scores)
            x_mean = x_extra

        mean_scores = (sorted_scores + sorted_scores_extra) / 2

        ax.plot(x_mean, mean_scores, label="mean")
        ax.legend()
    # ax.plot(x, [logit(y) for y in sorted_scores])
    ax.grid(linestyle="--")
    plt.xlabel("prob to have a score between 0 and y [%]")
    plt.ylabel("score")
    plt.show()

    percentiles = np.interp(np.linspace(0, 1, 20), x_mean / 100, mean_scores)
    for i, percentil in enumerate(percentiles):
        print("Top {}% have a score higher than {:.3f}".format(
            100 - (i * 5), percentil))

    np.save(open("data/percentiles.npy", "wb"), percentiles)

    # # fit polynomial to mean scores
    # ORDER = 3
    # coeffs = np.polyfit(x_mean, mean_scores, ORDER)
    # approx_scores = 0
    # for i, coeff in enumerate(reversed(coeffs)):
    #     approx_scores += coeff * x_mean ** i
    # plt.figure()
    # plt.plot(x_mean, mean_scores, label="mean")
    # plt.plot(x_mean, approx_scores, label="approx")
    # plt.legend()
    # plt.grid(linestyle="--")
    # plt.xlabel("prob to have a score between 0 and y")
    # plt.ylabel("score")
    # plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--db',
        dest='db_file',
        type=str
    )
    parser.add_argument(
        '--db-extra',
        dest='db_extra_file',
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logging.info('Called with args:')
    logging.info(args)
    main(args)
