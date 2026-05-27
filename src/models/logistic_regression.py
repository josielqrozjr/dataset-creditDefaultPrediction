"""Regressão Logística."""

from sklearn.linear_model import LogisticRegression
from config import RANDOM_SEED, HYPERPARAMS


def build_model():
    params = HYPERPARAMS["Logistic Regression"]
    return LogisticRegression(random_state=RANDOM_SEED, **params)
