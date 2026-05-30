"""K-Nearest Neighbors (Versão RAPIDS/GPU)."""

from cuml.neighbors import KNeighborsClassifier
from config import HYPERPARAMS

def build_model():
    # Usamos .copy() para garantir que não alteramos o dicionário global
    params = HYPERPARAMS.get("KNN", {}).copy()
    
    # Removemos o n_jobs, pois a aceleração paralela na GPU é nativa no cuML
    params.pop("n_jobs", None)
    
    return KNeighborsClassifier(**params)