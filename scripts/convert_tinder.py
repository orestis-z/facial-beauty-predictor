import os
import pickle
import argparse


def main(args):
    directories = os.listdir(args.db_dir)

    db = []
    for directory in directories:
        if directory[0] != ".":
            img_paths = []
            for sub_dir in os.listdir(os.path.join(args.db_dir, directory)):
                if sub_dir[0] != ".":
                    sub_sub_dir = os.path.join(
                        args.db_dir, directory, sub_dir, "processed_files")
                    files = os.listdir(sub_sub_dir)
                    file_640 = None
                    for file in files:
                        if "640" in file:
                            file_640 = file
                    if file_640 is not None:
                        img_paths.append(os.path.join(sub_sub_dir, file_640))
            db.append(dict(img_paths=img_paths))
    # db = db[:5]
    db_dict = dict(all=db)
    pickle.dump(db_dict, open(os.path.join(
        args.output_dir, "tinder.pkl"), "wb"))


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
