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
                "meta": dict(gender=row[0][1], ethnicity=row[0][0])
            } for row in rows}
            db_list.append(db)
    db_dict_all = db_list[0]
    db_dict_female = {key: el for key,
                      el in db_dict_all.items() if el["meta"]["gender"] == "F"}
    db_dict_male = {key: el for key,
                    el in db_dict_all.items() if el["meta"]["gender"] == "M"}
    db_dict_asian_female = {key: el for key,
                            el in db_dict_female.items() if el["meta"]["ethnicity"] == "A"}
    db_dict_caucasian_female = {key: el for key,
                                el in db_dict_female.items() if el["meta"]["ethnicity"] == "C"}
    db_dict_asian_male = {key: el for key,
                          el in db_dict_male.items() if el["meta"]["ethnicity"] == "A"}
    db_dict_caucasian_male = {key: el for key,
                              el in db_dict_male.items() if el["meta"]["ethnicity"] == "C"}
    db_dict = dict(data=dict(all=db_dict_all, train=db_list[1], test=db_list[2],
                             f=db_dict_female,
                             m=db_dict_male,
                             af=db_dict_asian_female,
                             am=db_dict_asian_male,
                             cf=db_dict_caucasian_female,
                             cm=db_dict_caucasian_male),
                   meta=dict(genders=("F", "M"), ethnicities=["A", "C"]))
    print("Stats:")
    print("Total size: {}".format(len(db_dict_all.keys())))
    print("Female size: {}".format(len(db_dict_female.keys())))
    print("Male size: {}".format(len(db_dict_male.keys())))
    print("Asian female size: {}".format(len(db_dict_asian_female.keys())))
    print("Asian male size: {}".format(len(db_dict_asian_male.keys())))
    print("Caucasian female size: {}".format(
        len(db_dict_caucasian_female.keys())))
    print("Caucasian male size: {}".format(len(db_dict_caucasian_male.keys())))
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
