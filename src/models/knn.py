"""K-Nearest Neighbors."""

from sklearn.neighbors import KNeighborsClassifier

from src.config import HYPERPARAMS


def build_model():
    params = HYPERPARAMS["KNN"]
    return KNeighborsClassifier(**params)
