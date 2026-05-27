"""Random Forest."""

from sklearn.ensemble import RandomForestClassifier
from config import RANDOM_SEED, HYPERPARAMS


def build_model():
    params = HYPERPARAMS["Random Forest"]
    return RandomForestClassifier(random_state=RANDOM_SEED, **params)
