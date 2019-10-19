import os
import sys
import argparse
import pickle

import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr
import sklearn.linear_model as linear_model
import sklearn.svm as svm
import sklearn.ensemble as ensemble
import sklearn.kernel_ridge as kernel_ridge
import sklearn.isotonic as isotonic
import sklearn.gaussian_process as gaussian_process
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import imageio
import joblib


def main(args):
    model_dir = os.path.join(
        args.output_dir, "models/{}".format("all/" if args.no_split else ""))
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    db_dict = pickle.load(open(args.db_file, "rb"))

    db_train = db_dict["all" if args.no_split else "train"]
    db_test = db_dict["test"]

    features = np.load(open(args.features, "rb"), allow_pickle=True).item()
    features_train = [features[profile_id]
                      for profile_id in db_train.keys()]
    features_test = [features[profile_id] for profile_id in db_test.keys()]
    nan_idx_train = np.argwhere(np.isnan(features_train))
    nan_idx_test = np.argwhere(np.isnan(features_test))
    nan_idx_train = np.unique(nan_idx_train[:, 0])
    nan_idx_test = np.unique(nan_idx_test[:, 0])
    print("Found {}/{} images without faces in train split".format(len(nan_idx_train), len(db_train)))
    print("Found {}/{} images without faces in test split".format(len(nan_idx_test), len(db_test)))
    features_train = np.delete(features_train, nan_idx_train, axis=0)
    features_test = np.delete(features_test, nan_idx_test, axis=0)
    db_flat_train = np.delete(list(db_train.values()), nan_idx_train)
    db_flat_test = np.delete(list(db_test.values()), nan_idx_test)

    scores_train = [d["score"] for d in db_flat_train]
    scores_test = [d["score"] for d in db_flat_test]

    # uncomment regressors to train
    model_info_list = (
        (linear_model.LinearRegression,),
        (linear_model.Lasso, dict(type="CONT", init=dict(alpha=1e-4))),
        (linear_model.Ridge, dict(type="CONT", init=dict(alpha=1e-4))),
        (linear_model.BayesianRidge, dict(type="CONT", init=dict(
            alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06))),

        # (linear_model.SGDRegressor,),
        # (linear_model.ElasticNet,),  # const
        # (linear_model.ARDRegression,),
        # (linear_model.HuberRegressor,),
        # (linear_model.Lars,),
        # (linear_model.LassoLars,),
        # (linear_model.PassiveAggressiveRegressor,),
        # (linear_model.TheilSenRegressor,),
        # (kernel_ridge.KernelRidge, dict(type="CONT", init=dict(
        #     alpha=1), kwargs=dict(kernel="sigmoid"))),
        # (svm.SVR,),
        # (ensemble.AdaBoostRegressor,),
        # (ensemble.GradientBoostingRegressor,),
        # (ensemble.RandomForestRegressor,),
        # (gaussian_process.GaussianProcessRegressor,
        # dict(type="CONT", init=dict(alpha=1e-10))),
    )
    for model_info in model_info_list:
        model_class = model_info[0]
        print("-" * 50)
        for i in range(args.max_order):
            order = i + 1
            print("Fitting {} w/ features of order {}".format(model_class.__name__, order))

            def get_model(kwargs={}):
                kwargs.update(meta.get("kwargs", {}))
                poly = PolynomialFeatures(order)
                model = model_class(*meta.get("args", {}),
                                    **kwargs)
                if args.pca:
                    pca = PCA(n_components=args.pca)
                    pipeline_list = [
                        ("poly", poly), ("pca", pca), ("fit", model)]
                else:
                    pipeline_list = [("poly", poly), ("fit", model)]
                return Pipeline(pipeline_list)

            if len(model_info) == 2:
                meta = model_info[1]
            else:
                meta = {}
            if meta.get("type") is None:
                print("Constant params")
                model = get_model()
                model.fit(features_train, scores_train)
            elif meta["type"] == "GRID":
                print("Finding optimal params from grid")
                param_grid = {"fit__" + k: v for k, v in meta["grid"].items()}
                model = get_model()
                model = GridSearchCV(model, param_grid=param_grid,).fit(
                    features_train, scores_train).best_estimator_
                print(model)
            elif meta["type"] == "CONT":
                print("Optimizing continuous params")
                init = meta["init"]

                def func(x):
                    kwargs = {k: x[i] for i, k in enumerate(init.keys())}
                    model = get_model(kwargs)
                    model.fit(features_train, scores_train)
                    scores_pred = model.predict(features_train)
                    pc = pearsonr(scores_pred, scores_train)[0]
                    return -pc
                res = minimize(func, list(init.values()), method="Nelder-Mead")
                print(res)
                res_kwargs = {k: res.x[i] for i, k in enumerate(init.keys())}
                model = model_class(**res_kwargs)
                model.fit(features_train, scores_train)
            else:
                raise ValueError

            if not args.no_split:
                scores_pred = model.predict(features_test)
                pc = pearsonr(scores_pred, scores_test)[0]
                print("PC: {:.3f}".format(pc))

            name = fullname(model_class)
            if args.pca:
                name += "PCA{}".format(args.pca)
            model_path = os.path.join(
                model_dir, "{}_{}.pkl".format(name, order))
            print("Saving model to {}".format(model_path))
            joblib.dump(model, model_path)


def fullname(cls):
    module = cls.__module__
    if module is None or module == str.__class__.__module__:
        return cls.__name__  # Avoid reporting __builtin__
    else:
        return module + "." + cls.__name__


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db",
        dest="db_file",
        type=str
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default="data",
        type=str
    )
    parser.add_argument(
        "--features",
        dest="features",
        default="data/mtcnn-facenet/features.npy",
        type=str
    )
    parser.add_argument(
        "--no-split",
        dest="no_split",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--pca",
        dest="pca",
        default=0,
        type=int
    )
    parser.add_argument(
        "--max-order",
        dest="max_order",
        default=2,
        type=int
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    main(args)
