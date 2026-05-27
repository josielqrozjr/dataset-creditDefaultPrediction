"""Stacked Generalization — base learners robustos e Logistic Regression como meta-learner."""

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from src.config import RANDOM_SEED, N_SPLITS
from src.models.random_forest import build_model as build_rf
from src.models.xgboost_model import build_model as build_xgb
from src.models.lightgbm_model import build_model as build_lgb

def build_model():
    estimators = [
        ("rf", build_rf()),
        ("xgb", build_xgb()),
        ("lgb", build_lgb()),
    ]
    
    # Meta-modelo configurado com balanceamento para corrigir possíveis vieses finais
    final_estimator = LogisticRegression(
        max_iter=1000, 
        solver="lbfgs", 
        class_weight="balanced", 
        random_state=RANDOM_SEED
    )
    
    return StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=N_SPLITS,
        n_jobs=-1,
        passthrough=False,
    )