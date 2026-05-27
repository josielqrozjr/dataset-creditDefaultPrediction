"""XGBoost."""

from xgboost import XGBClassifier

from src.config import RANDOM_SEED, HYPERPARAMS


def build_model():
    params = HYPERPARAMS["XGBoost"]
    return XGBClassifier(random_state=RANDOM_SEED, verbosity=0, **params)
