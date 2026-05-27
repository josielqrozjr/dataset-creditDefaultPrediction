"""Redes Neurais Artificiais (MLP)."""

from sklearn.neural_network import MLPClassifier
from config import RANDOM_SEED, HYPERPARAMS


def build_model():
    params = HYPERPARAMS["ANN (MLP)"]
    return MLPClassifier(random_state=RANDOM_SEED, **params)
