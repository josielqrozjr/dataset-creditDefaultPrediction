"""Redes Neurais Artificiais (MLP)."""

from sklearn.neural_network import MLPClassifier
from config import HYPERPARAMS


def build_model():
    params = HYPERPARAMS["ANN (MLP)"]
    return MLPClassifier(**params)
