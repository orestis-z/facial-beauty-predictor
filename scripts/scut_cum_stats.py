import os
import sys
import pickle
import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logit


def main(args):
    db_dict = pickle.load(open(args.db_file, "rb"))
    sorted_scores = sorted([d["score"] for _, d in db_dict["all"].items()])
    x = np.linspace(0, 1, len(sorted_scores))
    plt.plot(x, sorted_scores, ".", markersize=0.5)

    if args.db_extra_file:
        db_extra_dict = pickle.load(open(args.db_extra_file, "rb"))
        sorted_scores = sorted([d["score"]
                                for _, d in db_extra_dict["all"].items()])
        x = np.linspace(0, 1, len(sorted_scores))
        plt.plot(x, sorted_scores, ".", markersize=0.5)
    # plt.plot(x, [logit(y) for y in sorted_scores])
    plt.grid(linestyle="--")
    plt.xlabel("prob to have a score between 0 and y")
    plt.ylabel("score")
    plt.show()


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
