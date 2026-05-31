"""Random Forest."""

from sklearn.ensemble import RandomForestClassifier
from config import HYPERPARAMS


def build_model():
    params = HYPERPARAMS["Random Forest"]
    return RandomForestClassifier(**params)
