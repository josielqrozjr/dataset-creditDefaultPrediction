"""XGBoost (Aceleração Nativa em GPU / Compatível com RAPIDS)."""

from xgboost import XGBClassifier
from config import RANDOM_SEED, HYPERPARAMS

def build_model():
    # Usamos .copy() para garantir isolamento das variáveis de configuração
    params = HYPERPARAMS.get("XGBoost", {}).copy()
    
    # O XGBoost moderno aceita cuDF e CuPy nativamente. 
    # O direcionamento para a memória de vídeo (VRAM) já ocorre via 
    # parâmetros injetados pelo config.py (device='cuda').
    return XGBClassifier(random_state=RANDOM_SEED, verbosity=0, **params)