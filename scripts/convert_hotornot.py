import os
import pickle
import argparse
import csv

from config import SCORE_MARGIN


# db_dict = {split: {profile_id: {img_paths, score}}}


def main(args):
    db_list = []
    for i in range(5):
        split = i + 1
        with open(os.path.join(args.db_dir, "eccv2010_split{}.csv".format(split))) as f:
            rows = list(csv.reader(f, delimiter=','))
            scores = [float(row[1]) for row in rows]
            min_score = min(scores) - SCORE_MARGIN
            max_score = max(scores) + SCORE_MARGIN
            rows_train = [row for row in rows if row[2] == "train"]
            rows_test = [row for row in rows if row[2] == "test"]
            db_train = _rows_to_db(rows_train, min_score, max_score)
            db_test = _rows_to_db(rows_test, min_score, max_score)
            db_list.append((db_train, db_test))
    db_dict = {"train{}".format(i + 1): db[0] for i, db in enumerate(db_list)}
    db_dict.update({"test{}".format(i + 1): db[1]
                    for i, db in enumerate(db_list)})
    db_dict.update(dict(
        all={**db_dict["train1"], **db_dict["test1"]},
        train=db_dict["train1"],
        test=db_dict["test1"],
    ))
    pickle.dump(db_dict, open(os.path.join(
        args.output_dir, "hotornot.pkl"), "wb"))


def _rows_to_db(rows, min_score, max_score):
    return {row[0][:-4]: {
        "img_paths": [os.path.join(args.db_dir, "hotornot_face", row[0][:-3] + "jpg")],
        "score": (float(row[1]) - min_score) / (max_score - min_score)
    } for row in rows}


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
