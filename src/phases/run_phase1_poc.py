"""
Fase 1: Provas de Conceito (Validação Metodológica)
---------------------------------------------------
Este script executa dois experimentos empíricos fundamentais para o TCC:
1. Comprova a eficácia do Feature Selection (Base Completa vs Base Enxuta).
2. Comprova a superioridade do Balanceamento Algorítmico contra Undersampling.
"""

import sys
import time
import logging
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler

# Importações do nosso projeto
from config import RANDOM_SEED, RESULTS_DIR, DATA_PROCESSED
from src.evaluation.metrics import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

def run_dimensionality_poc(X_full: pd.DataFrame, X_reduced: pd.DataFrame, y: pd.Series):
    """Experimento 1: Maldição da Dimensionalidade e Feature Selection."""
    logger.info("=== Iniciando Experimento 1: Dimensionalidade ===")
    
    results = []
    datasets = {"Completa (3265 features)": X_full, "Enxuta (400 features)": X_reduced}
    
    # Instanciamos os modelos com balanceamento algorítmico já ativado
    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, n_jobs=-1, random_state=RANDOM_SEED),
        "XGBoost": XGBClassifier(scale_pos_weight=3, n_estimators=200, max_depth=6, n_jobs=-1, random_state=RANDOM_SEED)
    }

    for db_name, X_data in datasets.items():
        logger.info(f"-> Avaliando Base: {db_name}")
        
        # Split simples 80/20 para a prova de conceito
        X_train, X_val, y_train, y_val = train_test_split(X_data, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
        
        for model_name, model in models.items():
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            preds = model.predict_proba(X_val)[:, 1]
            metrics = evaluate_model(y_val, preds)
            
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

    return pd.DataFrame(results)


def run_balancing_poc(X_reduced: pd.DataFrame, y: pd.Series):
    """Experimento 2: Impacto das estratégias de tratamento de classe minoritária."""
    logger.info("=== Iniciando Experimento 2: Tratamento de Desbalanceamento ===")
    
    results = []
    
    # Split simples 80/20
    X_train, X_val, y_train, y_val = train_test_split(X_reduced, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)

    strategies = ["Sem Balanceamento", "Undersampling (Físico)", "Algorítmico (Cost-Sensitive)"]
    
    for strategy in strategies:
        logger.info(f"-> Avaliando Estratégia: {strategy}")
        
        # Ajuste dinâmico dos dados e hiperparâmetros com base na estratégia
        X_train_run, y_train_run = X_train, y_train
        lr_kwargs = {"max_iter": 1000, "n_jobs": -1, "random_state": RANDOM_SEED}
        xgb_kwargs = {"n_estimators": 200, "max_depth": 6, "n_jobs": -1, "random_state": RANDOM_SEED}

        if strategy == "Undersampling (Físico)":
            sampler = RandomUnderSampler(random_state=RANDOM_SEED)
            X_train_run, y_train_run = sampler.fit_resample(X_train, y_train)
        elif strategy == "Algorítmico (Cost-Sensitive)":
            lr_kwargs["class_weight"] = "balanced"
            xgb_kwargs["scale_pos_weight"] = 3
            # Se for "Sem Balanceamento", não passa parâmetros de peso (default)

        models = {
            "Logistic Regression": LogisticRegression(**lr_kwargs),
            "XGBoost": XGBClassifier(**xgb_kwargs)
        }

        for model_name, model in models.items():
            start_time = time.time()
            model.fit(X_train_run, y_train_run)
            train_time = time.time() - start_time
            
            preds = model.predict_proba(X_val)[:, 1]
            metrics = evaluate_model(y_val, preds)
            
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

    return pd.DataFrame(results)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Carregando bases de dados... (Isso pode demorar alguns minutos para a base completa)")
    
    # =========================================================================
    # ATENÇÃO: Ajuste os nomes dos arquivos conforme o seu pipeline salvou!
    # O ideal é usar o Polars para ler o parquet pesado e passar para Pandas.
    # =========================================================================
    try:
        # Exemplo de carregamento (Ajuste para o seu cenário real de I/O)
        # df_full = pl.read_parquet(DATA_PROCESSED / "train_aggregated.parquet").to_pandas()
        df_full = pd.read_parquet(DATA_PROCESSED / "train_aggregated.parquet") # Placeholder
        
        # Separa a variável alvo
        y = df_full["target"]
        X_full = df_full.drop(columns=["target"])
        
        # Carrega a lista de 400 features selecionadas
        with open(DATA_PROCESSED / "selected_features_list.txt", "r") as f:
            selected_cols = [line.strip() for line in f.readlines()]
            
        X_reduced = X_full[selected_cols]
        
    except FileNotFoundError as e:
        logger.error(f"Arquivo não encontrado: {e}")
        logger.error("Verifique se os dados agregados e a lista de features estão em 'data/processed/'")
        sys.exit(1)

    # 1. Roda a Prova de Dimensionalidade
    df_dim = run_dimensionality_poc(X_full, X_reduced, y)
    df_dim.to_csv(RESULTS_DIR / "poc_01_dimensionalidade.csv", index=False)
    
    # 2. Roda a Prova de Balanceamento (Usando a base reduzida)
    df_bal = run_balancing_poc(X_reduced, y)
    df_bal.to_csv(RESULTS_DIR / "poc_02_balanceamento.csv", index=False)
    
    logger.info("Fase 1 concluída com sucesso! Resultados exportados para a pasta 'results/'")


if __name__ == "__main__":
    main()