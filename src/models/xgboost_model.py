"""XGBoost."""

from xgboost import XGBClassifier
from config import HYPERPARAMS


def build_model():
    params = HYPERPARAMS["XGBoost"]
    return XGBClassifier(verbosity=0, **params)
