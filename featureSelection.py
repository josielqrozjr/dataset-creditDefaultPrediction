# ============================================================================
# FEATURE SELECTION PIPELINE
# ============================================================================

import os
import warnings
import logging
from datetime import datetime
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import duckdb

import matplotlib.pyplot as plt
import seaborn as sns
import shap

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# ============================================================================
# CONFIGURACAO GLOBAL
# ============================================================================
SEED = 42
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

CONFIG = {
    'correlation_threshold': 0.95,
    'missing_threshold': 0.90,
    'mutual_info_percentile': 75,
    'lgb_importance_percentile': 80,
    'seed': SEED,
}


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seeds(SEED)


# ============================================================================
# DETECCAO DE GPU
# ============================================================================
def check_cuda_available():
    try:
        if torch.cuda.is_available():
            return True
        torch.randn(1, device='cuda')
        return True
    except Exception:
        return False


GPU_AVAILABLE = check_cuda_available()

if GPU_AVAILABLE:
    try:
        DEVICE = torch.device("cuda")
    except Exception:
        DEVICE = torch.device("cpu")
        GPU_AVAILABLE = False
else:
    DEVICE = torch.device("cpu")

CONFIG['lightgbm_device'] = 'gpu' if GPU_AVAILABLE else 'cpu'


# ============================================================================
# LOGGING
# ============================================================================
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"feature_selection_{TIMESTAMP}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# MATPLOTLIB / SEABORN
# ============================================================================
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9


# ============================================================================
# DUCKDB
# ============================================================================
conn_duckdb = duckdb.connect(':memory:')


# ============================================================================
# PRINT DE AMBIENTE
# ============================================================================
print("=" * 80)
print("CONFIGURACAO DE AMBIENTE - FEATURE SELECTION PIPELINE")
print("=" * 80)
print(f"  Dispositivo: {DEVICE.type.upper()}")
print(f"  GPU Disponivel: {'SIM' if GPU_AVAILABLE else 'NAO'}")

if GPU_AVAILABLE:
    try:
        print(f"  Nome da GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except Exception as e:
        print(f"  Erro ao obter detalhes GPU: {e}")

print(f"  NumPy: {np.__version__}")
print(f"  Pandas: {pd.__version__}")
print(f"  LightGBM: {lgb.__version__}")
print(f"  SHAP: {shap.__version__}")
print(f"  Config: {CONFIG}")
print("=" * 80)

logger.info("INICIO DO PIPELINE DE FEATURE SELECTION")
logger.info(f"Device: {DEVICE}, GPU: {GPU_AVAILABLE}, Seed: {SEED}")


# ============================================================================
# ETAPA 1/5: LEITURA E PRE-PROCESSAMENTO
# ============================================================================
print("\n" + "=" * 80)
print("ETAPA 1/5: LEITURA E PRE-PROCESSAMENTO")
print("=" * 80)

# Caminhos dos dados (ajuste conforme necessario)
data_path = "./data/raw/parquet/train/data_*.parquet"
labels_path = "./data/raw/parquet/train_labels/data_*.parquet"

progress = tqdm(total=4, desc="Etapa 1/5 - Pre-processamento", unit="passo")

# --- 1.1 Leitura de dados ---
progress.set_postfix_str("Carregando dados")
try:
    df = conn_duckdb.execute(f"SELECT * FROM parquet_scan('{data_path}')").df()
    logger.info(f"Dados carregados via DuckDB: {data_path}")
except Exception as e:
    logger.warning(f"Fallback para pandas: {e}")
    df = pd.read_parquet(data_path)

try:
    labels = conn_duckdb.execute(f"SELECT * FROM parquet_scan('{labels_path}')").df()
    logger.info(f"Labels carregados via DuckDB: {labels_path}")
except Exception as e:
    logger.warning(f"Fallback para pandas (labels): {e}")
    labels = pd.read_parquet(labels_path)
progress.update(1)

# --- 1.2 Merge com labels ---
# train_labels possui colunas: customer_id, target
progress.set_postfix_str("Merge com labels")
target_col = 'target'

merge_col = 'customer_id'
if merge_col not in df.columns:
    raise ValueError(f"Coluna '{merge_col}' nao encontrada no dataset de treino. Colunas: {df.columns.tolist()[:20]}")
if merge_col not in labels.columns or target_col not in labels.columns:
    raise ValueError(f"Labels deve conter '{merge_col}' e '{target_col}'. Colunas: {labels.columns.tolist()}")

df = df.merge(labels[[merge_col, target_col]], on=merge_col, how='left')
logger.info(f"Merge realizado em '{merge_col}'. Shape apos merge: {df.shape}")

# Remover linhas sem target (caso left join nao encontre)
n_before = len(df)
df = df.dropna(subset=[target_col])
df[target_col] = df[target_col].astype(int)
n_after = len(df)
if n_before != n_after:
    logger.info(f"Removidas {n_before - n_after} linhas sem target")

progress.update(1)

# --- 1.3 Separacao target/features ---
progress.set_postfix_str("Separando target/features")
y = df[target_col].values
X = df.drop(columns=[target_col]).copy()

print(f"\n  Dataset: {df.shape[0]} linhas, {X.shape[1]} features")
print(f"  Target - Positivos: {y.sum()}/{len(y)} ({y.mean()*100:.2f}%)")
logger.info(f"X shape: {X.shape}, Target positivos: {y.sum()}/{len(y)}")
progress.update(1)

# --- 1.4 Remocao de features problematicas ---
progress.set_postfix_str("Removendo features problematicas")
features_to_remove = {}
final_features = X.columns.tolist()

# IDs (cardinalidade = n_amostras)
id_features = [col for col in X.columns if X[col].nunique() == len(X)]
for col in id_features:
    features_to_remove[col] = "ID (cardinalidade = n_amostras)"
final_features = [f for f in final_features if f not in id_features]

# Constantes
const_features = [col for col in final_features if X[col].nunique() == 1]
for col in const_features:
    features_to_remove[col] = "Constante"
final_features = [f for f in final_features if f not in const_features]

# Quase-constantes (>99% mesmo valor)
quasi_const_features = []
for col in final_features:
    value_counts = X[col].value_counts()
    if len(value_counts) > 0 and value_counts.iloc[0] / len(X) > 0.99:
        quasi_const_features.append(col)
        features_to_remove[col] = "Quase-constante (>99%)"
final_features = [f for f in final_features if f not in quasi_const_features]

# Missing > threshold
missing_features = []
for col in final_features:
    missing_pct = X[col].isna().sum() / len(X)
    if missing_pct > CONFIG['missing_threshold']:
        missing_features.append(col)
        features_to_remove[col] = f"Missing {missing_pct:.0%}"
final_features = [f for f in final_features if f not in missing_features]

X_clean = X[final_features].copy()

print(f"\n  Features removidas: {len(features_to_remove)}")
print(f"    - IDs: {len(id_features)}")
print(f"    - Constantes: {len(const_features)}")
print(f"    - Quase-constantes: {len(quasi_const_features)}")
print(f"    - Missing extremo: {len(missing_features)}")
print(f"  Features restantes: {X_clean.shape[1]}")

# Salvar relatorio de remocao
if features_to_remove:
    os.makedirs("./reports", exist_ok=True)
    removal_df = pd.DataFrame([
        {'Feature': feat, 'Motivo': reason}
        for feat, reason in features_to_remove.items()
    ])
    removal_df.to_csv(f"./reports/removed_features_{TIMESTAMP}.csv", index=False)

progress.update(1)
progress.close()
logger.info(f"Pre-processamento concluido: {X_clean.shape[1]} features restantes")


# ============================================================================
# ETAPA 2/5: CORRELACAO + MUTUAL INFORMATION
# ============================================================================
print("\n" + "=" * 80)
print("ETAPA 2/5: CORRELACAO + MUTUAL INFORMATION")
print("=" * 80)

progress = tqdm(total=4, desc="Etapa 2/5 - Correlacao + MI", unit="passo")

# --- 2.1 Preencher missing para analise ---
progress.set_postfix_str("Preenchendo missing")
X_filled = X_clean.fillna(X_clean.median(numeric_only=True))
progress.update(1)

# --- 2.2 Correlacao com target ---
progress.set_postfix_str("Calculando correlacoes")
pearson_corr = X_filled.corrwith(pd.Series(y, index=X_filled.index), method='pearson').abs()
pearson_corr = pearson_corr.sort_values(ascending=False)

spearman_corr = X_filled.corrwith(pd.Series(y, index=X_filled.index), method='spearman').abs()
spearman_corr = spearman_corr.sort_values(ascending=False)

print(f"\n  Top 10 Pearson com target:")
for feat, val in pearson_corr.head(10).items():
    print(f"    {feat}: {val:.4f}")

logger.info(f"Pearson top: {pearson_corr.index[0]} ({pearson_corr.iloc[0]:.4f})")
logger.info(f"Spearman top: {spearman_corr.index[0]} ({spearman_corr.iloc[0]:.4f})")
progress.update(1)

# --- 2.3 Remover features altamente correlacionadas entre si ---
progress.set_postfix_str("Removendo redundancias")

if X_filled.shape[0] > 100000:
    sample_idx = np.random.choice(X_filled.shape[0], 100000, replace=False)
    corr_matrix = X_filled.iloc[sample_idx].corr(method='pearson')
else:
    corr_matrix = X_filled.corr(method='pearson')

features_to_remove_corr = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > CONFIG['correlation_threshold']:
            feat1, feat2 = corr_matrix.columns[i], corr_matrix.columns[j]
            corr_target_1 = pearson_corr.get(feat1, 0)
            corr_target_2 = pearson_corr.get(feat2, 0)
            to_remove = feat2 if corr_target_1 >= corr_target_2 else feat1
            features_to_remove_corr.add(to_remove)

features_after_corr = [f for f in X_filled.columns if f not in features_to_remove_corr]
print(f"\n  Features removidas por alta correlacao (>{CONFIG['correlation_threshold']}): {len(features_to_remove_corr)}")
print(f"  Features restantes: {len(features_after_corr)}")
logger.info(f"Correlacao: removidas {len(features_to_remove_corr)} features")
progress.update(1)

# --- 2.4 Mutual Information ---
progress.set_postfix_str("Calculando Mutual Information")

n_classes = len(np.unique(y))
if n_classes < 50:
    mi_scores = mutual_info_classif(X_filled[features_after_corr], y, random_state=SEED)
    task_type = "Classificacao"
else:
    mi_scores = mutual_info_regression(X_filled[features_after_corr], y, random_state=SEED)
    task_type = "Regressao"

mi_scores = pd.Series(mi_scores, index=features_after_corr).sort_values(ascending=False)

mi_threshold = np.percentile(mi_scores, 100 - CONFIG['mutual_info_percentile'])
features_selected_corr_mi = mi_scores[mi_scores >= mi_threshold].index.tolist()

print(f"\n  Tipo de tarefa: {task_type}")
print(f"  Top 10 MI:")
for feat, val in mi_scores.head(10).items():
    print(f"    {feat}: {val:.4f}")
print(f"\n  Features apos filtro MI (top {CONFIG['mutual_info_percentile']}%): {len(features_selected_corr_mi)}")

logger.info(f"MI: {len(features_selected_corr_mi)} features selecionadas")
progress.update(1)
progress.close()

# --- Visualizacao MI ---
os.makedirs("./plots", exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 6))
top_mi = mi_scores.head(20)
ax.barh(range(len(top_mi)), top_mi.values)
ax.set_yticks(range(len(top_mi)))
ax.set_yticklabels(top_mi.index)
ax.set_xlabel("Mutual Information Score")
ax.set_title("Top 20 Features por Mutual Information")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"./plots/mutual_information_{TIMESTAMP}.png", dpi=300, bbox_inches='tight')
plt.close()

# Heatmap correlacao top features
top_n = min(30, len(features_selected_corr_mi))
if top_n > 1:
    top_feats = features_selected_corr_mi[:top_n]
    corr_top = X_filled[top_feats].corr(method='pearson')
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr_top, cmap='coolwarm', center=0, square=True,
                linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax, annot=False)
    ax.set_title(f"Matriz de Correlacao (Top {top_n} Features)")
    plt.tight_layout()
    plt.savefig(f"./plots/correlation_heatmap_{TIMESTAMP}.png", dpi=300, bbox_inches='tight')
    plt.close()

X_selected_mi = X_filled[features_selected_corr_mi].copy()


# ============================================================================
# ETAPA 3/5: LIGHTGBM FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 80)
print("ETAPA 3/5: LIGHTGBM FEATURE IMPORTANCE")
print("=" * 80)

progress = tqdm(total=4, desc="Etapa 3/5 - LightGBM", unit="passo")

# --- 3.1 Preparacao ---
progress.set_postfix_str("Preparando dados")
X_train = X_selected_mi.copy()
y_train = y.copy()

categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
if categorical_features:
    for col in categorical_features:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
    print(f"  {len(categorical_features)} features categoricas codificadas")

X_train = X_train.fillna(X_train.median(numeric_only=True))
progress.update(1)

# --- 3.2 Split ---
progress.set_postfix_str("Split train/valid")
stratify = y_train if n_classes < 50 else None
X_trn, X_val, y_trn, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=SEED, stratify=stratify
)
print(f"\n  Train: {X_trn.shape}, Valid: {X_val.shape}")
progress.update(1)

# --- 3.3 Treinamento ---
progress.set_postfix_str("Treinando LightGBM")

if n_classes == 2:
    objective = 'binary'
    metric = 'auc'
elif n_classes < 50:
    objective = 'multiclass'
    metric = 'multi_logloss'
else:
    objective = 'regression'
    metric = 'mse'

lgb_params = {
    'objective': objective,
    'metric': metric,
    'num_leaves': 255,
    'learning_rate': 0.05,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': SEED,
    'device': CONFIG['lightgbm_device'],
    'num_threads': -1,
}
if objective == 'multiclass':
    lgb_params['num_class'] = n_classes

train_data = lgb.Dataset(X_trn, label=y_trn, categorical_feature=categorical_features)
valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data,
                         categorical_feature=categorical_features)

booster = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data, valid_data],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)

print(f"  Modelo treinado: {booster.num_trees()} arvores")
logger.info(f"LightGBM: {booster.num_trees()} arvores")
progress.update(1)

# --- 3.4 Feature importance e selecao ---
progress.set_postfix_str("Extraindo importancia")

feature_importance = booster.feature_importance(importance_type='gain')
feature_imp_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importance,
    'Normalized_Importance': feature_importance / feature_importance.sum()
}).sort_values('Importance', ascending=False)

importance_threshold = np.percentile(
    feature_imp_df['Importance'],
    100 - CONFIG['lgb_importance_percentile']
)
features_final = feature_imp_df[
    feature_imp_df['Importance'] >= importance_threshold
]['Feature'].tolist()

print(f"\n  Top 15 features:")
for _, row in feature_imp_df.head(15).iterrows():
    print(f"    {row['Feature']:30s} | {row['Importance']:.4f}")

print(f"\n  Features com importancia = 0: {(feature_imp_df['Importance'] == 0).sum()}")
print(f"  Features selecionadas (top {CONFIG['lgb_importance_percentile']}%): {len(features_final)}")

# Validacao
if objective == 'binary':
    y_pred_val = booster.predict(X_val)
    val_auc = roc_auc_score(y_val, y_pred_val)
    print(f"  ROC-AUC (valid): {val_auc:.4f}")
    logger.info(f"ROC-AUC valid: {val_auc:.4f}")

# Salvar importancia
os.makedirs("./reports", exist_ok=True)
feature_imp_df.to_csv(f"./reports/lgb_importance_{TIMESTAMP}.csv", index=False)

progress.update(1)
progress.close()

# --- Visualizacoes LightGBM ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Histograma de importancia
axes[0].hist(feature_imp_df['Importance'], bins=50, edgecolor='black', alpha=0.7)
axes[0].axvline(importance_threshold, color='red', linestyle='--',
                label=f'Threshold: {importance_threshold:.4f}')
axes[0].set_xlabel("Feature Importance")
axes[0].set_ylabel("Frequencia")
axes[0].set_title("Distribuicao de Feature Importance")
axes[0].legend()

# Top 30
top_30 = feature_imp_df.head(30)
axes[1].barh(range(len(top_30)), top_30['Normalized_Importance'].values)
axes[1].set_yticks(range(len(top_30)))
axes[1].set_yticklabels(top_30['Feature'].values)
axes[1].set_xlabel("Importancia Normalizada")
axes[1].set_title("Top 30 Features")
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(f"./plots/lgb_importance_{TIMESTAMP}.png", dpi=300, bbox_inches='tight')
plt.close()

X_selected_lgb = X_train[features_final].copy()
booster_model = booster
logger.info(f"LightGBM feature selection: {len(features_final)} features")


# ============================================================================
# ETAPA 4/5: SHAP EXPLAINABILITY
# ============================================================================
print("\n" + "=" * 80)
print("ETAPA 4/5: SHAP EXPLAINABILITY")
print("=" * 80)

progress = tqdm(total=3, desc="Etapa 4/5 - SHAP", unit="passo")

# --- 4.1 Calcular SHAP values ---
progress.set_postfix_str("Calculando SHAP values")

sample_size = min(5000, len(X_val))
sample_idx = np.random.choice(len(X_val), sample_size, replace=False)
X_sample = X_val.iloc[sample_idx].reset_index(drop=True)

explainer = shap.TreeExplainer(booster_model)
shap_values = explainer.shap_values(X_sample)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

print(f"\n  SHAP calculado para {len(X_sample)} amostras")
logger.info(f"SHAP values shape: {shap_values.shape}")
progress.update(1)

# --- 4.2 Estatisticas SHAP ---
progress.set_postfix_str("Estatisticas SHAP")

shap_abs_mean = np.mean(np.abs(shap_values), axis=0)
shap_stats = pd.DataFrame({
    'Feature': X_sample.columns,
    'Mean_Abs_SHAP': shap_abs_mean,
}).sort_values('Mean_Abs_SHAP', ascending=False)

print(f"\n  Top 15 features por impacto SHAP:")
for _, row in shap_stats.head(15).iterrows():
    print(f"    {row['Feature']:30s} | {row['Mean_Abs_SHAP']:.6f}")

shap_stats.to_csv(f"./reports/shap_statistics_{TIMESTAMP}.csv", index=False)
progress.update(1)

# --- 4.3 Visualizacoes SHAP ---
progress.set_postfix_str("Gerando graficos SHAP")

# Summary beeswarm
shap.summary_plot(shap_values, X_sample, plot_type="beeswarm",
                  max_display=15, show=False)
plt.title("SHAP Summary Plot - Impacto das Features")
plt.tight_layout()
plt.savefig(f"./plots/shap_summary_beeswarm_{TIMESTAMP}.png", dpi=300, bbox_inches='tight')
plt.close()

# Summary bar
shap.summary_plot(shap_values, X_sample, plot_type="bar",
                  max_display=15, show=False)
plt.title("SHAP Feature Importance - Impacto Medio Absoluto")
plt.tight_layout()
plt.savefig(f"./plots/shap_summary_bar_{TIMESTAMP}.png", dpi=300, bbox_inches='tight')
plt.close()

# Dependence plots top 5
top_5_feats = shap_stats.head(5)['Feature'].tolist()
fig, axes = plt.subplots(len(top_5_feats), 1, figsize=(12, 4 * len(top_5_feats)))
if len(top_5_feats) == 1:
    axes = [axes]
for idx, feature in enumerate(top_5_feats):
    shap.dependence_plot(feature, shap_values, X_sample, ax=axes[idx], show=False)
    axes[idx].set_title(f"Dependencia SHAP: {feature}")
plt.tight_layout()
plt.savefig(f"./plots/shap_dependence_{TIMESTAMP}.png", dpi=300, bbox_inches='tight')
plt.close()

progress.update(1)
progress.close()
logger.info("SHAP explainability concluido")


# ============================================================================
# ETAPA 5/5: RESUMO FINAL
# ============================================================================
print("\n" + "=" * 80)
print("ETAPA 5/5: RESUMO FINAL")
print("=" * 80)

print(f"\n  REDUCAO DE FEATURES:")
print(f"    Iniciais:                  {X.shape[1]}")
print(f"    Apos pre-processamento:    {X_clean.shape[1]}")
print(f"    Apos Correlacao + MI:      {len(features_selected_corr_mi)}")
print(f"    Apos LightGBM Importance:  {len(features_final)}")
print(f"    Reducao total:             {((X.shape[1] - len(features_final)) / X.shape[1] * 100):.1f}%")

if objective == 'binary':
    print(f"\n  ROC-AUC (validacao): {val_auc:.4f}")

print(f"\n  FEATURES FINAIS ({len(features_final)}):")
for idx, feat in enumerate(features_final[:20], 1):
    imp = feature_imp_df[feature_imp_df['Feature'] == feat]['Importance'].values
    imp_str = f"{imp[0]:.4f}" if len(imp) > 0 else "N/A"
    print(f"    {idx:2d}. {feat:30s} | Importance: {imp_str}")
if len(features_final) > 20:
    print(f"    ... e mais {len(features_final) - 20} features")

# Salvar lista final
final_features_path = f"./reports/final_selected_features_{TIMESTAMP}.txt"
with open(final_features_path, 'w') as f:
    f.write(f"SELECTED FEATURES ({len(features_final)} total)\n")
    f.write("=" * 60 + "\n\n")
    for idx, feat in enumerate(features_final, 1):
        f.write(f"{idx}. {feat}\n")

print(f"\n  Artefatos salvos em:")
print(f"    - ./reports/")
print(f"    - ./plots/")
print(f"    - {log_file}")

print("\n" + "=" * 80)
print("PIPELINE CONCLUIDO - PRONTO PARA MODELAGEM")
print("=" * 80)

logger.info(f"PIPELINE COMPLETO - {len(features_final)} features selecionadas")
