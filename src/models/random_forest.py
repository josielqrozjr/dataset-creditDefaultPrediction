"""Random Forest (Versão RAPIDS/GPU)."""

from cuml.ensemble import RandomForestClassifier
from config import RANDOM_SEED, HYPERPARAMS

def build_model():
    # Usamos .copy() para garantir que não alteramos o dicionário global
    params = HYPERPARAMS.get("Random Forest", {}).copy()
    
    # -------------------------------------------------------------------------
    # Tratamento de compatibilidade: Scikit-Learn -> cuML
    # -------------------------------------------------------------------------
    # 1. n_jobs não existe (a GPU já paralela nativamente)
    params.pop("n_jobs", None)
    
    # 2. cuML não suporta balanceamento interno (class_weight='balanced') no RF
    params.pop("class_weight", None)
    
    # Garante o seed de reprodutibilidade
    params["random_state"] = RANDOM_SEED
    
    return RandomForestClassifier(**params)