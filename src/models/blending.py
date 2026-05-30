"""Blending — Ensemble com holdout explícito (20%) para treinar o meta-learner. (Versão RAPIDS/GPU)"""

import cupy as cp
import cudf
from sklearn.base import BaseEstimator, ClassifierMixin
from cuml.linear_model import LogisticRegression
from cuml.model_selection import train_test_split
from config import RANDOM_SEED
from src.models.random_forest import build_model as build_rf
from src.models.xgboost_model import build_model as build_xgb
from src.models.lightgbm_model import build_model as build_lgb

class BlendingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimators, meta_estimator, holdout_size=0.2, random_state=42):
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
        self.holdout_size = holdout_size
        self.random_state = random_state

    def fit(self, X, y):
        # Proteção contra formatos, convertendo para matrizes CuPy na VRAM
        X_arr = X.to_cupy() if hasattr(X, 'to_cupy') else cp.asarray(X, dtype=cp.float32)
        y_arr = y.to_cupy() if hasattr(y, 'to_cupy') else cp.asarray(y, dtype=cp.float32)

        # Split estratificado interno do Blending via cuML
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X_arr, y_arr, 
            test_size=self.holdout_size, 
            stratify=y_arr, 
            random_state=self.random_state
        )

        # Alocação da matriz de meta-features diretamente na GPU
        meta_features_holdout = cp.zeros((X_holdout.shape[0], len(self.base_estimators)), dtype=cp.float32)

        # Treina a base no subset e gera predições no holdout
        for i, (name, model) in enumerate(self.base_estimators):
            model.fit(X_train, y_train)
            
            # Extrai as probabilidades da classe positiva
            preds = model.predict_proba(X_holdout)
            # Proteção para modelos que possam retornar matriz 1D no modo binário
            meta_features_holdout[:, i] = preds[:, 1] if len(preds.shape) > 1 else preds

        # O Meta-learner aprende a combinar baseado apenas nas predições do holdout
        self.meta_estimator.fit(meta_features_holdout, y_holdout)

        # Retreinar base learners em todo o dataset (X) para não perder dados em produção
        for name, model in self.base_estimators:
            model.fit(X_arr, y_arr)

        return self

    def predict_proba(self, X):
        X_arr = X.to_cupy() if hasattr(X, 'to_cupy') else cp.asarray(X, dtype=cp.float32)
        meta_features = cp.zeros((X_arr.shape[0], len(self.base_estimators)), dtype=cp.float32)
        
        for i, (name, model) in enumerate(self.base_estimators):
            preds = model.predict_proba(X_arr)
            meta_features[:, i] = preds[:, 1] if len(preds.shape) > 1 else preds
            
        return self.meta_estimator.predict_proba(meta_features)

    def predict(self, X):
        probs = self.predict_proba(X)
        probs_positive = probs[:, 1] if len(probs.shape) > 1 else probs
        return (probs_positive >= 0.5).astype(cp.int32)

def build_model():
    base_estimators = [
        ("rf", build_rf()),
        ("xgb", build_xgb()),
        ("lgb", build_lgb()),
    ]
    
    # Meta-modelo na GPU utilizando cuML. O L-BFGS é coberto nativamente pelo solver QN do cuML.
    meta_estimator = LogisticRegression(
        max_iter=1000, 
    )
    
    return BlendingClassifier(
        base_estimators=base_estimators,
        meta_estimator=meta_estimator,
        holdout_size=0.2,
        random_state=RANDOM_SEED,
    )