import os
import pickle
import argparse


def main(args):
    db_list = []
    for path in ("All_labels.txt", "split_of_60%training and 40%testing/train.txt", "split_of_60%training and 40%testing/test.txt"):
        with open(os.path.join(args.db_dir, "train_test_files", path)) as f:
            lines = f.read().splitlines()
            db = [line.split(" ") for line in lines]
            db = [{
                "img_paths": [os.path.join(args.db_dir, "Images", d[0])],
                "score": float(d[1]) / 5,  # normalize [0, 1]
                "width": 350,
                "height": 350} for d in db]
            # db = db[:5]
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
        os.mkdir(args.output_dir)
    main(args)
