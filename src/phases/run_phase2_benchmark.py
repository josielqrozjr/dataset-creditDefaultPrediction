"""
Fase 2: Campeonato Aberto (Baseline dos Modelos)
------------------------------------------------
Avalia os 7 algoritmos base individuais utilizando a base enxuta (400 features).
Gera as predições Out-Of-Fold (OOF) via Validação Cruzada Estratificada para
estabelecer o ranking justo e identificar o Top 3 para a Fase 3.
"""

import sys
import time
import logging
import gc
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Importações do projeto
from config import RANDOM_SEED, RESULTS_DIR, TRAIN_DATA_PATH, SELECTED_FEATURES_PATH, N_SPLITS
from src.evaluation.metrics import evaluate_model
from src.models.registry import MODEL_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

# Definimos os 7 modelos individuais (excluímos os Ensembles que ficam para a Fase 4)
BASE_MODELS = [
    "Logistic Regression",
    "KNN",
    "ANN (MLP)",
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "CatBoost"
]

def load_and_prepare_data():
    """Carrega os dados e aplica o Feature Selection isolando as 400 features."""
    logger.info("Passo 1: Carregando a base de treino via Polars (Otimizado para RAM)...")
    df_pd = pl.scan_parquet(TRAIN_DATA_PATH).collect().to_pandas()
    
    logger.info("Passo 2: Removendo colunas de texto (IDs e Datas)...")
    cols_to_drop = [col for col in ["customer_ID", "S_2"] if col in df_pd.columns]
    if cols_to_drop:
        df_pd = df_pd.drop(columns=cols_to_drop)
        
    object_cols = df_pd.select_dtypes(include=['object', 'string', 'category']).columns
    if len(object_cols) > 0:
        df_pd = df_pd.drop(columns=object_cols)

    logger.info("Passo 3: Separando Alvo e convertendo para float32...")
    y = df_pd["target"].astype("int8")
    X_full = df_pd.drop(columns=["target"]).astype("float32")
    
    del df_pd
    gc.collect()
    
    logger.info("Passo 4: Aplicando máscara do Feature Selection (Base Enxuta)...")
    with open(SELECTED_FEATURES_PATH, "r") as f:
        selected_cols = [line.strip() for line in f.readlines()]
        if "target" in selected_cols:
            selected_cols.remove("target")
        selected_cols = [col for col in selected_cols if col in X_full.columns]
        
    X_reduced = X_full[selected_cols]
    
    del X_full
    gc.collect()
    
    logger.info(f"Dados prontos! Shape final para o Campeonato: {X_reduced.shape}")
    return X_reduced, y

def wrap_model_if_needed(model_name, model_instance):
    """
    Algoritmos baseados em árvores modernas lidam com NaNs nativamente.
    Algoritmos clássicos do Scikit-Learn quebram com NaNs e exigem imputação.
    """
    modern_algorithms = ["XGBoost", "LightGBM", "CatBoost"]
    
    if model_name in modern_algorithms:
        return model_instance
    else:
        # Empacota modelos sensíveis em um pipeline automático
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", model_instance)
        ])

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        X, y = load_and_prepare_data()
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        sys.exit(1)
        
    # Inicializa Validação Cruzada Estratificada
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    
    results = []
    
    logger.info(f"=== INICIANDO CAMPEONATO ABERTO ({len(BASE_MODELS)} Modelos) ===")
    
    # Loop principal: Batalha dos Modelos
    for model_name in BASE_MODELS:
        if model_name not in MODEL_REGISTRY:
            logger.warning(f"Modelo {model_name} não encontrado no Registry. Pulando...")
            continue
            
        logger.info(f"\n-> Treinando: {model_name}")
        start_time = time.time()
        
        # Instancia o modelo puro do registro
        raw_model = MODEL_REGISTRY[model_name]()
        model = wrap_model_if_needed(model_name, raw_model)
        
        # Array para guardar predições Out-Of-Fold
        oof_preds = np.zeros(len(X))
        
        fold_metrics_amex = []
        
        # Executa K-Fold
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            
            # Treinamento
            model.fit(X_train, y_train)
            
            # Predição (Pegando probabilidade da classe 1)
            preds = model.predict_proba(X_val)
            preds_positive = preds[:, 1] if len(preds.shape) > 1 else preds
            
            oof_preds[val_idx] = preds_positive
            
            # Avalia fold localmente apenas para logar a evolução
            metrics = evaluate_model(y.iloc[val_idx], preds_positive)
            fold_metrics_amex.append(metrics["AMEX_Score"])
            
            logger.info(f"   Fold {fold+1}/{N_SPLITS} | AMEX: {metrics['AMEX_Score']:.4f}")
            
            gc.collect()
            
        total_time = time.time() - start_time
        
        # Avalia a predição OOF inteira consolidadamente (Simula performance em produção)
        oof_metrics = evaluate_model(y, oof_preds)
        
        logger.info(f"-> [RESULTADO FINAL - {model_name}]")
        logger.info(f"   AMEX Global (OOF): {oof_metrics['AMEX_Score']:.4f} | Tempo Total: {total_time:.1f}s")
        
        results.append({
            "Modelo": model_name,
            "Tempo Total (s)": round(total_time, 2),
            "Tempo por Fold (s)": round(total_time / N_SPLITS, 2),
            "AMEX Score (OOF)": oof_metrics["AMEX_Score"],
            "ROC AUC (OOF)": oof_metrics["ROC_AUC"],
            "AUPRC (OOF)": oof_metrics["AUPRC"],
            "F1-Score (OOF)": oof_metrics["F1_Score"],
            "Recall (OOF)": oof_metrics["Recall"]
        })
        
    # Exporta ranking
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="AMEX Score (OOF)", ascending=False).reset_index(drop=True)
    
    output_path = RESULTS_DIR / "phase2_benchmark_ranking.csv"
    df_results.to_csv(output_path, index=False)
    
    logger.info("\n=========================================")
    logger.info(f"Fase 2 Concluída! Ranking salvo em: {output_path}")
    logger.info("Top 3 Modelos Recomendados para a Fase 3 (Optuna):")
    for i, row in df_results.head(3).iterrows():
        logger.info(f" {i+1}º - {row['Modelo']} (AMEX: {row['AMEX Score (OOF)']:.4f})")
    logger.info("=========================================")

if __name__ == "__main__":
    main()