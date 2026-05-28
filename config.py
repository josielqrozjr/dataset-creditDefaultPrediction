"""
Configurações Globais e Hiperparâmetros do Benchmark AMEX
---------------------------------------------------------
Este arquivo centraliza todos os caminhos de diretório, sementes de 
reprodutibilidade, parâmetros padrão dos modelos (Fase 2) e os espaços 
de busca do Optuna (Fase 3).
"""

from pathlib import Path

# =====================================================================
# 1. Diretórios e Caminhos
# =====================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_BEST_MODELS = PROJECT_ROOT / "results_best_models"

# Arquivos de Dados Específicos
TRAIN_DATA_PATH = DATA_PROCESSED / "merge_split" / "train_80.parquet"
TEST_DATA_PATH = DATA_PROCESSED / "merge_split" / "valid_20.parquet"
SELECTED_FEATURES_PATH = DATA_PROCESSED / "selection" / "selected_features_list.txt"

# =====================================================================
# 2. Reprodutibilidade e Divisão de Dados
# =====================================================================
RANDOM_SEED = 42
N_SPLITS = 5  # Número de partições do StratifiedKFold (Fase 2 e 3)
TEST_SIZE = 0.20  # 20% da base total isolada para o teste cego final

# =====================================================================
# 3. Hiperparâmetros Padrão (Fase 2 - Campeonato Aberto)
# Nota: O balanceamento algorítmico já está embutido nestas configurações.
# =====================================================================
HYPERPARAMS = {
    "Logistic Regression": {
        "max_iter": 1000,
        "solver": "lbfgs",
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
    },
    "KNN": {
        "n_neighbors": 5,
        "n_jobs": -1,
    },
    "ANN (MLP)": {
        "hidden_layer_sizes": (64, 32),
        "max_iter": 300,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "random_state": RANDOM_SEED,
    },
    "Random Forest": {
        "n_estimators": 150,
        "max_depth": 15,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    },
    "LightGBM": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "is_unbalance": True,  # Balanceamento nativo do LGBM
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    },
    "XGBoost": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "scale_pos_weight": 3,  # Proxy de balanceamento (75% / 25%)
        "eval_metric": "logloss",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    },
    "CatBoost": {
        "iterations": 200,
        "depth": 6,
        "learning_rate": 0.1,
        "auto_class_weights": "Balanced", # Balanceamento nativo do CatBoost
        "random_seed": RANDOM_SEED,
        "verbose": 0,
    }
}

# =====================================================================
# 4. Espaços de Busca do Optuna (Fase 3 - Otimização Suprema)
# Focado exclusivamente nos algoritmos mais robustos (Boosting).
# =====================================================================
OPTUNA_GRIDS = {
    "LightGBM": {
        "n_estimators": ("int", 100, 500),
        "max_depth": ("int", 3, 12),
        "num_leaves": ("int", 20, 150),
        "learning_rate": ("float", 0.01, 0.2, "log"),
        "min_child_samples": ("int", 10, 100),
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.5, 1.0),
    },
    "XGBoost": {
        "n_estimators": ("int", 100, 500),
        "max_depth": ("int", 3, 10),
        "learning_rate": ("float", 0.01, 0.2, "log"),
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.5, 1.0),
        "min_child_weight": ("int", 1, 10),
    },
    "CatBoost": {
        "iterations": ("int", 100, 500),
        "depth": ("int", 4, 10),
        "learning_rate": ("float", 0.01, 0.2, "log"),
        "l2_leaf_reg": ("float", 1e-3, 10.0, "log"),
        "border_count": ("int", 32, 255),
    }
}