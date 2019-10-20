# Attractiveness Estimator

## Based on

[tindetheus](https://github.com/cjekel/tindetheus)

[FaceNet](https://github.com/davidsandberg/facenet)

[MTCNN](https://github.com/ipazc/mtcnn)

## Requirements:

- python 3.7
- [pipenv](https://github.com/pypa/pipenv)

## Installation

`pipenv install --dev`

## Quick Start

- Download [SCUT](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release) dataset
- Download [HotOrNot](https://www.researchgate.net/publication/261595808_Female_Facial_Beauty_Dataset_ECCV2010_v10) dataset
- Download the FaceNet model [20170512-110547](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk/edit) and extract it into the `data` directory
- Convert datasets with:
  - `python scripts/convert_scut.py --db-dir <path/to/db/dir>`
  - `python scripts/convert_tinder.py --db-dir <path/to/db/dir>`
  - `python scripts/convert_hotornot.py --db-dir <path/to/db/dir>`
- Generate features for SCUT dataset once and store them to the disk:
  - `python scripts/generate_features_async.py --db data/scut.pkl --output-dir data/scut`
- Train regressor models with
  - `python scripts/train_regressor.py --db data/scut.pkl --output-dir data/scut/mtcnn-facenet --features data/scut/mtcnn-facenet/features.npy`
- Compare regressors:
  - `python scripts/compare_models.py --db data/scut.pkl --output-dir data/scut/mtcnn-facenet --models-dir data/scut/mtcnn-facenet/models --features data/scut/mtcnn-facenet/features.npy`
- Generate model trained on all the dataset:
  - `python scripts/train_regressor.py --db data/scut.pkl --output-dir data/scut/mtcnn-facenet --features data/scut/mtcnn-facenet/features.npy --no-split`
- Generate features for Tinder dataset once and store them to the disk:
  - `python scripts/generate_features_async.py --db data/tinder.pkl --output-dir data/tinder`
- Infer results on tinder dataset:
  - `python scripts/infer.py --db data/tinder.pkl --features data/tinder/mtcnn-facenet/features.npy --model data/scut/mtcnn-facenet/models/all/sklearn.linear_model.base.LinearRegression_1.pkl`

Those steps can be repeated for a mtcnn-only backbone (put `--backbone mtcnn` flag where necessary and replace `mtcnn-facenet` with `mtcnn`)

## Results

### SCUT

FaceNet features:

| Regressor | PC    |
| --------- | ----- |
| Lasso     | 0.846 |
| Ridge     | 0.872 |
| Linear    | 0.872 |

FaceNet + MTCNN features:

@TODO (note: was slightly better than Facenet features only)

MTCNN only features:

@TODO

## HotOrNot

FaceNet features:

| Regressor | PC    |
| --------- | ----- |
| Linear    | 0.536 |
| Lasso     | 0.550 |
| Ridge     | 0.567 |

FaceNet + MTCNN features:

@TODO

MTCNN only features:

@TODO
