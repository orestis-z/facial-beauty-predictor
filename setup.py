from setuptools import setup

setup(
    name='attractiveness_estimator',
    packages=['attractiveness_estimator'],
    install_requires=[
        "numpy >= 1.11.3",
        "matplotlib >= 2.0.0",
        "imageio >= 2.2.0",
        "scikit-learn >= 0.19.0",
        "scikit-image >= 0.13.1",
        "tensorflow >= 1.15.0,<2.0",
        "joblib >= 0.13.0",
        "opencv-python >= 4.1.1",
    ],
)
