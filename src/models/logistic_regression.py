"""Regressão Logística."""

from sklearn.linear_model import LogisticRegression
from config import HYPERPARAMS


def build_model():
    params = HYPERPARAMS["Logistic Regression"]
    # Passamos apenas o **params, pois o config já tem o random_state!
    return LogisticRegression(**params)