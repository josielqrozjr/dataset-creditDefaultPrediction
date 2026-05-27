"""Voting Classifier — integração de XGBoost, LightGBM e CatBoost com soft voting."""

from sklearn.ensemble import VotingClassifier
from src.models.xgboost_model import build_model as build_xgb
from src.models.lightgbm_model import build_model as build_lgb
from src.models.catboost_model import build_model as build_cat

def build_model():
    estimators = [
        ("xgb", build_xgb()),
        ("lgb", build_lgb()),
        ("cat", build_cat()),
    ]
    # O soft voting é vital pois a Métrica AMEX depende das probabilidades preditas
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)