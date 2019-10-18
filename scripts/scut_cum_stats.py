import os
import sys
import pickle
import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logit


def main(args):
    data = pickle.load(open(args.data_file, "rb"))

    sorted_scores = sorted([d["score"] for d in data["all"]])

    x = np.arange(0, 1, 1 / len(sorted_scores))

    plt.plot(x, sorted_scores)
    # plt.plot(x, [logit(y) for y in sorted_scores])
    plt.grid(linestyle="--")
    plt.xlabel("prob to have a score between 0 and y")
    plt.ylabel("score")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        dest='data_file',
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
