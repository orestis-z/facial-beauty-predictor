class Config(object):
    FLASK_ENV_SHORTNAME = "none"
    DEBUG = False
    TESTING = False

    LOG_METRICS = True
    LOG_METRICS_TIMEOUT = 5 * 60  # [s] == 5 min

    FACENET_MODEL_PATH = "s3://facial-beauty-predictor/facenet/20170512-110547"
    REGRESSOR_MODEL_PATH = "s3://facial-beauty-predictor/regressor_model.pkl"
    PERCENTILS_PATH = "s3://facial-beauty-predictor/percentiles.npy"


class DevelopmentConfig(Config):
    FLASK_ENV_SHORTNAME = "dev"
    DEBUG = True

    # LOG_METRICS_TIMEOUT = 1 * 5  # [s]

    # REGRESSOR_MODEL_PATH = "data/scut/mtcnn-facenet/models/sklearn.linear_model.base.LinearRegression_1.pkl"
    PERCENTILS_PATH = "data/percentiles.npy"


class StagingConfig(Config):
    FLASK_ENV_SHORTNAME = "stage"
    DEBUG = True


class ProductionConfig(Config):
    FLASK_ENV_SHORTNAME = "prod"
