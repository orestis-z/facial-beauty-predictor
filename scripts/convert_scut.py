import os
import pickle
import argparse


from config import SCORE_MARGIN


# db_dict = {split: {profile_id: {img_paths, score}}}


def main(args):
    db_list = []
    for path in ("All_labels.txt", "split_of_60%training and 40%testing/train.txt", "split_of_60%training and 40%testing/test.txt"):
        with open(os.path.join(args.db_dir, "train_test_files", path)) as f:
            lines = f.read().splitlines()
            rows = [line.split(" ") for line in lines]
            scores = [float(row[1]) for row in rows]
            min_score = min(scores) - SCORE_MARGIN
            max_score = max(scores) + SCORE_MARGIN
            db = {row[0][:-4]: {
                "img_paths": [os.path.join(args.db_dir, "Images", row[0])],
                # normalize [0, 1]
                "score": (float(row[1]) - min_score) / (max_score - min_score),
            } for row in rows}
            db_list.append(db)
    db_dict = dict(all=db_list[0], train=db_list[1], test=db_list[2])
    pickle.dump(db_dict, open(os.path.join(
        args.output_dir, "scut.pkl"), "wb"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--db-dir',
        dest='db_dir',
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        default='data',
        type=str
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main(args)
