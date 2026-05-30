"""CatBoost (Compatibilidade RAPIDS/GPU)."""

import cupy as cp
from catboost import CatBoostClassifier
from config import RANDOM_SEED, HYPERPARAMS

class RAPIDSCatBoost:
    """
    Wrapper para garantir que o CatBoost funcione no pipeline RAPIDS.
    Embora o motor do CatBoost treine na GPU (task_type='GPU'), sua API Python 
    possui limitações na leitura direta de DataFrames do cuDF ou CuPy.
    Este wrapper faz a ponte de I/O em tempo real, mantendo a compatibilidade.
    """
    def __init__(self, **kwargs):
        self.model = CatBoostClassifier(**kwargs)
        self.classes_ = None

    def _to_cpu(self, data):
        """Converte cuDF ou CuPy para Pandas/NumPy (CPU) temporariamente."""
        if hasattr(data, 'to_pandas'):
            return data.to_pandas()
        elif hasattr(data, 'get'):  # Método do CuPy para transferir para NumPy
            return data.get()
        return data

    def fit(self, X, y):
        X_cpu = self._to_cpu(X)
        y_cpu = self._to_cpu(y)
        
        self.model.fit(X_cpu, y_cpu)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, X):
        X_cpu = self._to_cpu(X)
        probs = self.model.predict_proba(X_cpu)
        
        # Devolve a matriz como CuPy para continuar no fluxo da VRAM
        return cp.asarray(probs, dtype=cp.float32)

    def predict(self, X):
        X_cpu = self._to_cpu(X)
        preds = self.model.predict(X_cpu)
        return cp.asarray(preds, dtype=cp.int32)

def build_model():
    params = HYPERPARAMS.get("CatBoost", {}).copy()
    
    # Garantias de configuração para o wrapper
    params["random_seed"] = RANDOM_SEED
    params["task_type"] = "GPU"
    params["verbose"] = 0
    
    return RAPIDSCatBoost(**params)