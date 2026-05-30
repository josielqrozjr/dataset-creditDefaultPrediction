"""
Fase 1: Provas de Conceito (Validação Metodológica) - Versão RAPIDS/GPU
-----------------------------------------------------------------------
Este script executa dois experimentos empíricos fundamentais para o TCC:
1. Comprova a eficácia do Feature Selection (Base Completa vs Base Enxuta).
2. Comprova a superioridade do Balanceamento Algorítmico contra Undersampling.

Toda a manipulação de dados e treinamento ocorre nativamente na VRAM.
"""

import sys
import time
import logging
import gc
import pandas as pd
import cudf
import cupy as cp
from pathlib import Path

from cuml.model_selection import train_test_split
from cuml.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Importações do nosso projeto
from config import RANDOM_SEED, RESULTS_DIR, TRAIN_DATA_PATH, SELECTED_FEATURES_PATH, GPU_AVAILABLE, DEVICE
from src.evaluation.metrics import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

def gpu_random_undersampler(X: cudf.DataFrame, y: cudf.Series, random_state: int = 42):
    """
    Função customizada para realizar Undersampling diretamente na GPU,
    já que o imblearn não suporta aceleração por hardware.
    """
    df = X.copy()
    df['target'] = y
    
    minority = df[df['target'] == 1]
    majority = df[df['target'] == 0]
    
    # Amostra a classe majoritária para igualar o tamanho da minoritária
    majority_sampled = majority.sample(n=len(minority), random_state=random_state)
    
    # Concatena e embaralha os dados
    undersampled = cudf.concat([minority, majority_sampled]).sample(frac=1.0, random_state=random_state)
    
    return undersampled.drop(columns=['target']), undersampled['target']


def run_dimensionality_poc(X_full: cudf.DataFrame, X_reduced: cudf.DataFrame, y: cudf.Series):
    """Experimento 1: Maldição da Dimensionalidade e Feature Selection."""
    logger.info("=== Iniciando Experimento 1: Dimensionalidade ===")
    
    results = []
    datasets = {"Completa (3265 features)": X_full, "Enxuta (400 features)": X_reduced}
    
    xgb_params = {"scale_pos_weight": 3, "n_estimators": 200, "max_depth": 6, "random_state": RANDOM_SEED}
    xgb_params["tree_method"] = "hist"
    xgb_params["device"] = "cuda"

    models = {
        # O cuML utiliza o solver Quasi-Newton (qn) no lugar do lbfgs
        "Logistic Regression": LogisticRegression(max_iter=500, solver='qn'),
        "XGBoost": XGBClassifier(**xgb_params)
    }

    for db_name, X_data in datasets.items():
        logger.info(f"-> Preparando split para a base: {db_name}")
        X_train, X_val, y_train, y_val = train_test_split(X_data, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
        
        logger.info("-> Split concluído. Limpando lixo da memória de vídeo...")
        gc.collect()
        
        for model_name, model in models.items():
            logger.info(f"-> [{model_name}] Treinando...")
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Extração de probabilidades. O cuML pode retornar um array 1D no modo binário.
            preds = model.predict_proba(X_val)
            preds_positive = preds[:, 1] if len(preds.shape) > 1 else preds
            
            metrics = evaluate_model(y_val, preds_positive)
            
            results.append({
                "Experimento": "Dimensionalidade",
                "Modelo": model_name,
                "Base de Dados": db_name,
                "Tempo Treino (s)": round(train_time, 2),
                "AMEX Score": metrics["AMEX_Score"],
                "ROC AUC": metrics["ROC_AUC"],
                "AUPRC": metrics["AUPRC"]
            })
            logger.info(f"   [{model_name}] Tempo: {train_time:.1f}s | AMEX: {metrics['AMEX_Score']:.4f}")
            
            gc.collect()

    return pd.DataFrame(results)


def run_balancing_poc(X_reduced: cudf.DataFrame, y: cudf.Series):
    """Experimento 2: Impacto das estratégias de tratamento de classe minoritária."""
    logger.info("=== Iniciando Experimento 2: Tratamento de Desbalanceamento ===")
    
    results = []
    
    X_train, X_val, y_train, y_val = train_test_split(X_reduced, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)

    strategies = ["Sem Balanceamento", "Undersampling (Físico)", "Algorítmico (Cost-Sensitive)"]
    
    for strategy in strategies:
        logger.info(f"-> Avaliando Estratégia: {strategy}")
        
        X_train_run, y_train_run = X_train, y_train
        lr_kwargs = {"max_iter": 500, "solver": "qn"}
        xgb_kwargs = {"n_estimators": 200, "max_depth": 6, "random_state": RANDOM_SEED, "tree_method": "hist", "device": "cuda"}

        if strategy == "Undersampling (Físico)":
            X_train_run, y_train_run = gpu_random_undersampler(X_train, y_train, random_state=RANDOM_SEED)
        elif strategy == "Algorítmico (Cost-Sensitive)":
            # Nota técnica: cuML Logistic Regression não suporta class_weight nativamente.
            # O peso do experimento recairá no XGBoost para validar a hipótese metodológica.
            xgb_kwargs["scale_pos_weight"] = 3

        models = {
            "Logistic Regression": LogisticRegression(**lr_kwargs),
            "XGBoost": XGBClassifier(**xgb_kwargs)
        }

        for model_name, model in models.items():
            start_time = time.time()
            model.fit(X_train_run, y_train_run)
            train_time = time.time() - start_time
            
            preds = model.predict_proba(X_val)
            preds_positive = preds[:, 1] if len(preds.shape) > 1 else preds
            
            metrics = evaluate_model(y_val, preds_positive)
            
            results.append({
                "Experimento": "Balanceamento",
                "Modelo": model_name,
                "Estratégia": strategy,
                "Tempo Treino (s)": round(train_time, 2),
                "AMEX Score": metrics["AMEX_Score"],
                "AUPRC": metrics["AUPRC"],
                "Recall": metrics["Recall"]
            })
            logger.info(f"   [{model_name}] Tempo: {train_time:.1f}s | AMEX: {metrics['AMEX_Score']:.4f} | Recall: {metrics['Recall']:.4f}")
            gc.collect()

    return pd.DataFrame(results)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Passo 1: Carregando base de dados nativamente na GPU com cuDF...")
    try:
        df_full = cudf.read_parquet(TRAIN_DATA_PATH)
        
        logger.info("Passo 2: Removendo colunas de texto (Prevenção de Erro 'String to Float')...")
        cols_to_drop = [col for col in ["customer_ID", "S_2"] if col in df_full.columns]
        if cols_to_drop:
            df_full = df_full.drop(columns=cols_to_drop)
            
        # Garante que qualquer outra string que tenha sobrado seja removida verificando os dtypes
        object_cols = [col for col, dtype in df_full.dtypes.items() if dtype in ['object', 'string', 'category']]
        if len(object_cols) > 0:
            df_full = df_full.drop(columns=object_cols)

        logger.info("Passo 3: Convertendo tipos para float32 (Economia de VRAM)...")
        y = df_full["target"].astype("int8")
        X_full = df_full.drop(columns=["target"]).astype("float32")
        
        del df_full
        gc.collect()
        
        logger.info("Passo 4: Carregando lista de features da base enxuta...")
        with open(SELECTED_FEATURES_PATH, "r") as f:
            selected_cols = [line.strip() for line in f.readlines()]
            if "target" in selected_cols:
                selected_cols.remove("target")
            
            selected_cols = [col for col in selected_cols if col in X_full.columns]
            
        X_reduced = X_full[selected_cols]
        logger.info(f"Formato -> Completa: {X_full.shape} | Enxuta: {X_reduced.shape}")
        
    except FileNotFoundError as e:
        logger.exception(f"Arquivo não encontrado: {e.filename}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Erro fatal no processamento dos dados: {e}")
        sys.exit(1)

    # 1. Roda a Prova de Dimensionalidade
    df_dim = run_dimensionality_poc(X_full, X_reduced, y)
    df_dim.to_csv(RESULTS_DIR / "poc_01_dimensionalidade.csv", index=False)
    
    logger.info("Deletando Base Completa (3265 features) da VRAM definitivamente...")
    del X_full
    gc.collect()
    
    # 2. Roda a Prova de Balanceamento (Usando a base reduzida)
    df_bal = run_balancing_poc(X_reduced, y)
    df_bal.to_csv(RESULTS_DIR / "poc_02_balanceamento.csv", index=False)
    
    logger.info("Fase 1 concluída com sucesso! Resultados exportados para a pasta 'results/'")


if __name__ == "__main__":
    main()