# Attractiveness Estimator

## Based on

[tindetheus](https://github.com/cjekel/tindetheus)

[FaceNet](https://github.com/davidsandberg/facenet)

[MTCNN](https://github.com/ipazc/mtcnn)

## Requirements:

- python3

## Installation

`python3 -m venv venv`

`. venv/bin/activate`

`python setup.py develop`

## Quick Start

- Download [SCUT](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release) database
- Download the FaceNet model [20170512-110547](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk/edit) and extract it into the `data` directory
- Convert databases with:
  - `python scripts/convert_scut.py --db-dir <path/to/db/dir>`
  - `python scripts/convert_tinder.py --db-dir <path/to/db/dir>`
- Generate features for SCUT dataset once and store them to the disk:
  - `python scripts/generate_features_async.py --db data/scut.pkl`
- Train regressor models with
  - `python scripts/train_regressor.py --db data/scut.pkl --output-dir data/mtcnn-facenet --features-train data/mtcnn-facenet/features-train.npy --features-test data/mtcnn-facenet/features-test.npy`
- Compare regressors:
  - `python scripts/compare_models.py --db data/scut.pkl --output-dir data/mtcnn-facenet --models-dir data/mtcnn-facenet/models --features data/mtcnn-facenet/features-test.npy`
- Generate model trained on all the dataset:
  - `python scripts/train_regressor.py --db data/scut.pkl --output-dir data/mtcnn-facenet --features-train data/mtcnn-facenet/features-all.npy --no-split`
- Generate features for Tinder dataset once and store them to the disk:
  - `python scripts/generate_features_async.py --db data/tinder.pkl --output-dir data/tinder --split all`
- Infer results on tinder dataset:
  - `python scripts/infer.py --db data/tinder.pkl --features data/tinder/mtcnn-facenet/features-all.npy --model data/mtcnn-facenet/models/all/sklearn.linear_model.base.LinearRegression_1.pkl`

Those steps can be repeated for a mtcnn-only backbone (put `--backbone mtcnn` flag where necessary and replace `mtcnn-facenet` with `mtcnn`)

## TODO

- Train on ECCV HotOrNot [dataset](https://www.researchgate.net/publication/261595808_Female_Facial_Beauty_Dataset_ECCV2010_v10)
- 
