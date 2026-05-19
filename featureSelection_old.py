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
        test_tensor = torch.randn(1, device='cuda')
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

print("=" * 80)
print("CONFIGURACAO DE AMBIENTE - FEATURE SELECTION PIPELINE")
print("=" * 80)
print(f"\n  Dispositivo: {DEVICE.type.upper()}")
print(f"  GPU Disponivel: {'SIM' if GPU_AVAILABLE else 'NAO'}")

if GPU_AVAILABLE:
    try:
        print(f"  Nome da GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Numero de GPUs: {torch.cuda.device_count()}")
        print(f"  Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except Exception as e:
        print(f"⚠ Erro ao obter detalhes GPU: {e}")
else:
    print("⚠ GPU não detectada. Usando CPU (operações serão mais lentas)")

print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"LightGBM version: {lgb.__version__}")
print(f"SHAP version: {shap.__version__}")

# ============================================================================
# 3. CONFIGURAÇÃO DE LOGGING
# ============================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f"feature_selection_{timestamp}.log")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("="*80)
logger.info("INÍCIO DO PIPELINE DE FEATURE SELECTION")
logger.info("="*80)
logger.info(f"Device: {DEVICE}")
logger.info(f"GPU Available: {GPU_AVAILABLE}")
logger.info(f"Seed: {SEED}")

# ============================================================================
# 4. INICIALIZAÇÃO DE DUCKDB
# ============================================================================
# Usar DuckDB em-memória para operações rápidas
conn_duckdb = duckdb.connect(':memory:')
logger.info("DuckDB inicializado em memória")

# ============================================================================
# 5. CONFIGURAÇÃO DE MATPLOTLIB E SEABORN
# ============================================================================
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configurar tamanho padrão de figuras
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

logger.info("Matplotlib e Seaborn configurados")

# ============================================================================
# 6. VARIÁVEIS GLOBAIS DE CONFIGURAÇÃO
# ============================================================================
# Thresholds e parâmetros do pipeline
CONFIG = {
    'correlation_threshold': 0.95,  # Para remover features altamente correlacionadas
    'missing_threshold': 0.90,       # Features com > 90% missing serão removidas
    'mutual_info_percentile': 75,    # Top 75% para seleção por MI
    'lgb_importance_percentile': 80, # Top 80% para seleção por LightGBM
    'lightgbm_device': 'gpu' if GPU_AVAILABLE else 'cpu',
    'seed': SEED,
}

print(f"\n✓ Configurações do pipeline:")
for key, value in CONFIG.items():
    print(f"  - {key}: {value}")

logger.info(f"Pipeline config: {CONFIG}")
print("\n" + "="*80)
print("AMBIENTE PRONTO PARA EXECUÇÃO")
print("="*80)


# ============================================================================
# CÉLULA 2: LEITURA E PRÉ-PROCESSAMENTO INICIAL
# ============================================================================

# ============================================================================
# 1. LEITURA DE DADOS (AJUSTE O CAMINHO CONFORME NECESSÁRIO)
# ============================================================================
# IMPORTANTE: Substitua os paths conforme seu ambiente
data_path = r"C:\Users\joaov\Workspace\4 ano\tcc\ml-credit-default-prediction\data\raw\parquet\train\data_*.parquet"  # Ajuste conforme necessário
labels_path = r"C:\Users\joaov\Workspace\4 ano\tcc\ml-credit-default-prediction\data\raw\parquet\train_labels\data_*.parquet"

# Opção 1: Ler com DuckDB (recomendado para grandes volumes)
try:
    df = conn_duckdb.execute(f"SELECT * FROM parquet_scan('{data_path}')").df()
    logger.info(f"Dados carregados via DuckDB: {data_path}")
except Exception as e:
    # Fallback para pandas se DuckDB falhar
    logger.warning(f"Fallback para pandas: {e}")
    df = pd.read_parquet(data_path)

# Carregar labels
try:
    labels = conn_duckdb.execute(f"SELECT * FROM parquet_scan('{labels_path}')").df()
    logger.info(f"Labels carregados via DuckDB: {labels_path}")
except Exception as e:
    logger.warning(f"Fallback para pandas (labels): {e}")
    labels = pd.read_parquet(labels_path)

# Merge com labels (ajuste as colunas conforme sua base)
# IMPORTANTE: Adapte os nomes de colunas (ex: 'customer_id' pode variar)
try:
    df = df.merge(labels, on=['customer_id', 'month_id'], how='left')
    target_col = 'target'  # Ajuste conforme necessário
    logger.info("Labels mesclados ao dataset")
except Exception as e:
    logger.warning(f"Merge automático falhou: {e}")
    target_col = 'target'  # Defina manualmente se necessário

print("=" * 80)
print("PRÉ-PROCESSAMENTO INICIAL")
print("=" * 80)
print(f"\n📊 Dataset Original:")
print(f"  Shape: {df.shape}")
print(f"  Colunas: {df.columns.tolist()[:10]}... (total: {len(df.columns)})")
print(f"  Memory: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

# ============================================================================
# 2. SEPARAÇÃO DE TARGET E FEATURES
# ============================================================================
# Identificar coluna de target
if target_col not in df.columns:
    logger.error(f"Target '{target_col}' não encontrado. Colunas: {df.columns.tolist()}")
    raise ValueError(f"Target '{target_col}' não encontrado")

y = df[[target_col]].copy()
X = df.drop(columns=[target_col]).copy()

logger.info(f"Target shape: {y.shape}, Features shape: {X.shape}")
print(f"\n🎯 Target:")
print(f"  Distribuição: {y[target_col].value_counts().to_dict()}")
print(f"  Taxa de positivos: {(y[target_col].sum() / len(y) * 100):.2f}%")

# ============================================================================
# 3. IDENTIFICAÇÃO DE FEATURES PROBLEMÁTICAS
# ============================================================================
features_to_remove = {}
final_features = X.columns.tolist()

print(f"\n🔍 Análise de Features Problemáticas:")

# 3.1 REMOVER IDs (alta cardinalidade)
print(f"\n  [1] Identificando IDs (cardinalidade = n_amostras)...")
id_features = []
for col in X.columns:
    if X[col].nunique() == len(X):
        id_features.append(col)
        features_to_remove[col] = "ID (cardinalidade = n_amostras)"

if id_features:
    print(f"      Removidas {len(id_features)} IDs: {id_features[:5]}")
    logger.info(f"IDs removidas: {id_features}")
else:
    print(f"      Nenhum ID encontrado")

final_features = [f for f in final_features if f not in id_features]

# 3.2 REMOVER CONSTANTES (variância = 0)
print(f"\n  [2] Identificando features constantes...")
const_features = []
for col in final_features:
    if X[col].nunique() == 1:
        const_features.append(col)
        features_to_remove[col] = "Constante (único valor)"

if const_features:
    print(f"      Removidas {len(const_features)} constantes: {const_features}")
    logger.info(f"Features constantes removidas: {const_features}")
else:
    print(f"      Nenhuma feature constante encontrada")

final_features = [f for f in final_features if f not in const_features]

# 3.3 REMOVER QUASE-CONSTANTES (variância muito baixa)
print(f"\n  [3] Identificando features quase-constantes...")
quasi_const_features = []
for col in final_features:
    try:
        # Calcular razão da classe mais frequente
        value_counts = X[col].value_counts()
        if len(value_counts) > 0:
            freq_ratio = value_counts.iloc[0] / len(X)
            if freq_ratio > 0.99:  # 99% da mesma classe
                quasi_const_features.append(col)
                features_to_remove[col] = f"Quase-constante (freq={freq_ratio:.2%})"
    except Exception as e:
        logger.debug(f"Erro ao calcular frequência de {col}: {e}")

if quasi_const_features:
    print(f"      Removidas {len(quasi_const_features)} quase-constantes")
    logger.info(f"Features quase-constantes: {quasi_const_features}")
else:
    print(f"      Nenhuma feature quase-constante encontrada")

final_features = [f for f in final_features if f not in quasi_const_features]

# 3.4 REMOVER FEATURES COM MISSING > THRESHOLD
print(f"\n  [4] Identificando features com missing > {CONFIG['missing_threshold']*100:.0f}%...")
missing_features = []
missing_stats = {}

for col in final_features:
    missing_pct = X[col].isna().sum() / len(X)
    missing_stats[col] = missing_pct
    
    if missing_pct > CONFIG['missing_threshold']:
        missing_features.append(col)
        features_to_remove[col] = f"Missing > {CONFIG['missing_threshold']*100:.0f}% ({missing_pct:.2%})"

if missing_features:
    print(f"      Removidas {len(missing_features)} features com missing extremo")
    for feat in missing_features[:5]:
        print(f"        - {feat}: {missing_stats[feat]:.2%} missing")
    logger.info(f"Features com missing extremo: {missing_features}")
else:
    print(f"      Nenhuma feature com missing extremo")

final_features = [f for f in final_features if f not in missing_features]

# ============================================================================
# 4. DATASET APÓS PRÉ-PROCESSAMENTO
# ============================================================================
X_clean = X[final_features].copy()

print(f"\n" + "="*80)
print(f"📈 Resumo do Pré-Processamento:")
print(f"="*80)
print(f"\n  Features iniciais: {X.shape[1]}")
print(f"  Features removidas: {X.shape[1] - X_clean.shape[1]}")
print(f"  Features finais: {X_clean.shape[1]}")
print(f"  Redução: {((X.shape[1] - X_clean.shape[1]) / X.shape[1] * 100):.1f}%")

# ============================================================================
# 5. RELATÓRIO DETALHADO DE FEATURES REMOVIDAS
# ============================================================================
if features_to_remove:
    print(f"\n📋 Features Removidas:")
    removal_df = pd.DataFrame([
        {'Feature': feat, 'Motivo': reason}
        for feat, reason in features_to_remove.items()
    ])
    print(removal_df.to_string(index=False))
    
    # Salvar relatório
    report_path = f"./reports/removed_features_{timestamp}.csv"
    os.makedirs("./reports", exist_ok=True)
    removal_df.to_csv(report_path, index=False)
    logger.info(f"Relatório de features removidas salvo em: {report_path}")

# ============================================================================
# 6. ESTATÍSTICAS DE MISSING
# ============================================================================
print(f"\n📊 Top 10 Features com Maior Missing (remanescentes):")
missing_remaining = {col: missing_stats.get(col, X_clean[col].isna().sum() / len(X_clean)) 
                     for col in X_clean.columns}
missing_sorted = sorted(missing_remaining.items(), key=lambda x: x[1], reverse=True)[:10]
for feat, pct in missing_sorted:
    print(f"  {feat}: {pct:.2%}")

# ============================================================================
# 7. GUARDAR VARIÁVEIS PARA PRÓXIMAS CÉLULAS
# ============================================================================
print(f"\n✓ Dataset preparado para feature selection")
logger.info(f"Dataset pronto: X_clean.shape={X_clean.shape}, y.shape={y.shape}")
print("="*80)

# ============================================================================
# CÉLULA 3: CORRELAÇÃO + MUTUAL INFORMATION
# ============================================================================

print("="*80)
print("ANÁLISE DE CORRELAÇÃO + MUTUAL INFORMATION")
print("="*80)

# ============================================================================
# 1. TRATAMENTO DE MISSING (necessário para correlação)
# ============================================================================
print(f"\n[1] Preparando dados para análise de correlação...")

# Estratégia: preencher missing com mediana (conservador)
X_filled = X_clean.fillna(X_clean.median(numeric_only=True))
y_target = y[target_col].values

logger.info(f"Missing preenchido com mediana. Shape final: {X_filled.shape}")
print(f"    ✓ Missing preenchido para análise")

# ============================================================================
# 2. CÁLCULO DE CORRELAÇÃO DE PEARSON
# ============================================================================
print(f"\n[2] Calculando Correlação de Pearson...")

# Usar DuckDB para cálculo otimizado de correlações
# Criar tabela temporária no DuckDB
conn_duckdb.register("features_df", X_filled)

# Calcular correlação com target usando numpy (DuckDB não otimiza isso bem)
pearson_corr = X_filled.corrwith(y_target, method='pearson').abs()
pearson_corr = pearson_corr.sort_values(ascending=False)

print(f"    ✓ Pearson calculado para {len(pearson_corr)} features")
print(f"\n    Top 10 features por correlação de Pearson:")
print(pearson_corr.head(10).to_string())

logger.info(f"Pearson correlation computed: top={pearson_corr.index[0]} ({pearson_corr.iloc[0]:.4f})")

# ============================================================================
# 3. CÁLCULO DE CORRELAÇÃO DE SPEARMAN
# ============================================================================
print(f"\n[3] Calculando Correlação de Spearman...")

spearman_corr = X_filled.corrwith(y_target, method='spearman').abs()
spearman_corr = spearman_corr.sort_values(ascending=False)

print(f"    ✓ Spearman calculado para {len(spearman_corr)} features")
print(f"\n    Top 10 features por correlação de Spearman:")
print(spearman_corr.head(10).to_string())

logger.info(f"Spearman correlation computed: top={spearman_corr.index[0]} ({spearman_corr.iloc[0]:.4f})")

# ============================================================================
# 4. IDENTIFICAR FEATURES ALTAMENTE CORRELACIONADAS ENTRE SI
# ============================================================================
print(f"\n[4] Identificando redundância entre features...")

# Calcular matriz de correlação entre features (apenas para sample, se muitos dados)
if X_filled.shape[0] > 100000:
    sample_idx = np.random.choice(X_filled.shape[0], 100000, replace=False)
    corr_matrix = X_filled.iloc[sample_idx].corr(method='pearson')
    print(f"    ⚠ Dataset grande: usando sample de 100k amostras para matriz de correlação")
else:
    corr_matrix = X_filled.corr(method='pearson')

# Encontrar pares altamente correlacionados
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > CONFIG['correlation_threshold']:
            feat1, feat2 = corr_matrix.columns[i], corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            
            # Guardar: feature com menor correlação com target (será removida)
            corr_target_1 = pearson_corr.get(feat1, 0)
            corr_target_2 = pearson_corr.get(feat2, 0)
            
            to_remove = feat2 if corr_target_1 > corr_target_2 else feat1
            to_keep = feat1 if corr_target_1 > corr_target_2 else feat2
            
            high_corr_pairs.append({
                'Feature_Removida': to_remove,
                'Feature_Mantida': to_keep,
                'Correlacao': corr_val,
                'Corr_Target_Removida': corr_target_1 if to_remove == feat1 else corr_target_2,
                'Corr_Target_Mantida': corr_target_2 if to_remove == feat1 else corr_target_1
            })

if high_corr_pairs:
    print(f"    ✗ {len(high_corr_pairs)} pares altamente correlacionados (|r| > {CONFIG['correlation_threshold']})")
    high_corr_df = pd.DataFrame(high_corr_pairs)
    print(f"\n    {high_corr_df.head(10).to_string(index=False)}")
    
    # Features a remover
    features_to_remove_corr = high_corr_df['Feature_Removida'].unique().tolist()
    logger.info(f"High correlation pairs found: {len(high_corr_pairs)}")
    logger.info(f"Features to remove (high correlation): {features_to_remove_corr}")
else:
    print(f"    ✓ Nenhum par com |r| > {CONFIG['correlation_threshold']}")
    features_to_remove_corr = []

# ============================================================================
# 5. CÁLCULO DE MUTUAL INFORMATION
# ============================================================================
print(f"\n[5] Calculando Mutual Information (MI) com target...")

# Determinar se é classificação ou regressão
if y_target.dtype in ['int64', 'int32', 'uint8'] or len(np.unique(y_target)) < 50:
    mi_scores = mutual_info_classif(X_filled, y_target, random_state=SEED)
    task_type = "Classificação"
    logger.info(f"Task type: Classification (n_classes={len(np.unique(y_target))})")
else:
    mi_scores = mutual_info_regression(X_filled, y_target, random_state=SEED)
    task_type = "Regressão"
    logger.info(f"Task type: Regression")

mi_scores = pd.Series(mi_scores, index=X_filled.columns)
mi_scores = mi_scores.sort_values(ascending=False)

print(f"    ✓ MI calculado ({task_type})")
print(f"\n    Top 10 features por Mutual Information:")
print(mi_scores.head(10).to_string())

# Visualizar distribuição de MI
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
mi_scores_sorted = mi_scores.sort_values(ascending=False)
ax.barh(range(min(20, len(mi_scores_sorted))), mi_scores_sorted.head(20).values)
ax.set_yticks(range(min(20, len(mi_scores_sorted))))
ax.set_yticklabels(mi_scores_sorted.head(20).index)
ax.set_xlabel("Mutual Information Score")
ax.set_title("Top 20 Features por Mutual Information")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"./plots/mutual_information_{timestamp}.png", dpi=300, bbox_inches='tight')
plt.show()

logger.info(f"MI plot saved")

# ============================================================================
# 6. SELEÇÃO COMBINADA: CORRELAÇÃO + MI
# ============================================================================
print(f"\n[6] Seleção Combinada (Correlação + MI)...")

# Features remanescentes após remover redundâncias
features_after_corr = [f for f in X_filled.columns if f not in features_to_remove_corr]

# Selecionar top features por MI (percentile)
mi_threshold = np.percentile(mi_scores[features_after_corr], 
                              100 - CONFIG['mutual_info_percentile'])
selected_mi = mi_scores[mi_scores >= mi_threshold].index.tolist()

# Features selecionadas
features_selected_corr_mi = list(set(features_after_corr) & set(selected_mi))
features_selected_corr_mi.sort(key=lambda x: mi_scores[x], reverse=True)

print(f"    ✓ Features selecionadas: {len(features_selected_corr_mi)}")
print(f"      - Após remover correlações: {len(features_after_corr)}")
print(f"      - Após filtro de MI (top {CONFIG['mutual_info_percentile']}%): {len(features_selected_corr_mi)}")

logger.info(f"Features after correlation filtering: {len(features_after_corr)}")
logger.info(f"Features after MI filtering: {len(features_selected_corr_mi)}")

# ============================================================================
# 7. VISUALIZAÇÃO: HEATMAP DE CORRELAÇÃO (TOP FEATURES)
# ============================================================================
print(f"\n[7] Gerando visualizações...")

# Selecionar top features para heatmap (por performance)
top_n = min(30, len(features_selected_corr_mi))
top_features = features_selected_corr_mi[:top_n]

if len(top_features) > 1:
    corr_top = X_filled[top_features].corr(method='pearson')
    
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr_top, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax, annot=False)
    ax.set_title(f"Matriz de Correlação (Top {top_n} Features)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"./plots/correlation_heatmap_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()
    logger.info(f"Correlation heatmap saved")

# ============================================================================
# 8. RESUMO E ESTATÍSTICAS
# ============================================================================
print(f"\n" + "="*80)
print(f"📊 RESUMO - CORRELAÇÃO + MUTUAL INFORMATION")
print(f"="*80)
print(f"\nFeatures Iniciais:                  {len(X_filled.columns)}")
print(f"Features Removidas (Alta Correlação): {len(features_to_remove_corr)}")
print(f"Features Selecionadas:              {len(features_selected_corr_mi)}")
print(f"Redução:                            {((len(X_filled.columns) - len(features_selected_corr_mi)) / len(X_filled.columns) * 100):.1f}%")
print(f"\n✓ Pipeline de Correlação + MI concluído")
print("="*80)

# ============================================================================
# 9. GUARDAR VARIÁVEIS PARA PRÓXIMAS CÉLULAS
# ============================================================================
# Estas variáveis serão usadas na Célula 4
X_selected_mi = X_filled[features_selected_corr_mi].copy()

logger.info(f"Correlation + MI filtering complete: {len(features_selected_corr_mi)} features remaining")

# ============================================================================
# CÉLULA 4: LIGHTGBM FEATURE IMPORTANCE
# ============================================================================

print("="*80)
print("FEATURE IMPORTANCE - LIGHTGBM")
print("="*80)

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# ============================================================================
# 1. PREPARAÇÃO DOS DADOS
# ============================================================================
print(f"\n[1] Preparando dados para treinamento LightGBM...")

X_train = X_selected_mi.copy()
y_train = y_target.copy()

# Tratar features categóricas (se houver)
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
if categorical_features:
    print(f"    ⚠ {len(categorical_features)} features categóricas encontradas")
    le_dict = {}
    for col in categorical_features:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        le_dict[col] = le
    logger.info(f"Categorical features encoded: {categorical_features}")
else:
    print(f"    ✓ Todas as features são numéricas")

# Validação final
assert not X_train.isna().any().any(), "X_train contém NaN após preparação"
print(f"    ✓ Dados validados: shape={X_train.shape}")

# ============================================================================
# 2. SPLIT TRAIN/VALID COM STRATIFICAÇÃO
# ============================================================================
print(f"\n[2] Dividindo dados (80/20 com estratificação)...")

from sklearn.model_selection import train_test_split

X_trn, X_val, y_trn, y_val = train_test_split(
    X_train, y_train, 
    test_size=0.2, 
    random_state=SEED,
    stratify=y_train if len(np.unique(y_train)) < 50 else None
)

print(f"    ✓ Train: {X_trn.shape}")
print(f"    ✓ Valid: {X_val.shape}")
print(f"    ✓ Target distribution - Train: {np.bincount(y_trn.astype(int))}")

# ============================================================================
# 3. CONFIGURAÇÃO E TREINAMENTO LIGHTGBM
# ============================================================================
print(f"\n[3] Treinando modelo LightGBM...")

# Detectar tipo de problema
n_classes = len(np.unique(y_train))
if n_classes == 2:
    objective = 'binary'
    metric = 'auc'
    is_classification = True
else:
    objective = 'multiclass'
    metric = 'multi_logloss'
    is_classification = True

# Parâmetros otimizados para feature selection
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
    'device': CONFIG['lightgbm_device'],  # 'gpu' ou 'cpu'
    'num_threads': -1,
}

if objective == 'multiclass':
    lgb_params['num_class'] = n_classes

# Criar datasets LightGBM
train_data = lgb.Dataset(X_trn, label=y_trn, categorical_feature=categorical_features)
valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data, 
                         categorical_feature=categorical_features)

# Treinar modelo
booster = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data, valid_data],
    valid_names=['train', 'valid'],
    early_stopping_rounds=50,
    verbose_eval=50,
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
)

print(f"\n    ✓ Modelo treinado: {booster.num_trees()} árvores")
logger.info(f"LightGBM model trained: {booster.num_trees()} trees")

# ============================================================================
# 4. EXTRAÇÃO DE FEATURE IMPORTANCE
# ============================================================================
print(f"\n[4] Extraindo Feature Importance...")

# Obter importância por Gain (redução de loss)
feature_importance = booster.feature_importance(importance_type='gain')
feature_names = X_train.columns

# Criar DataFrame
feature_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance,
    'Normalized_Importance': feature_importance / feature_importance.sum()
})

feature_imp_df = feature_imp_df.sort_values('Importance', ascending=False)

print(f"    ✓ Importância extraída")
print(f"\n    Top 15 features por LightGBM:")
print(feature_imp_df.head(15).to_string(index=False))

# Salvar importância
importance_path = f"./reports/lgb_importance_{timestamp}.csv"
os.makedirs("./reports", exist_ok=True)
feature_imp_df.to_csv(importance_path, index=False)
logger.info(f"Feature importance saved to {importance_path}")

# ============================================================================
# 5. ANÁLISE DE FEATURES INÚTEIS
# ============================================================================
print(f"\n[5] Identificando features com baixa importância...")

# Features com importância próxima de zero
zero_importance = feature_imp_df[feature_imp_df['Importance'] == 0]
low_importance_threshold = np.percentile(feature_imp_df['Importance'], 5)
low_importance = feature_imp_df[feature_imp_df['Importance'] < low_importance_threshold]

print(f"    ✗ Features com importância = 0: {len(zero_importance)}")
print(f"    ✗ Features com importância < {low_importance_threshold:.6f} (5º percentil): {len(low_importance)}")

if len(zero_importance) > 0:
    print(f"\n    Features com importância nula:")
    print(zero_importance[['Feature', 'Importance']].head(10).to_string(index=False))
    logger.info(f"Zero importance features: {zero_importance['Feature'].tolist()}")

# ============================================================================
# 6. SELEÇÃO FINAL: TOP N% FEATURES
# ============================================================================
print(f"\n[6] Seleção Final de Features...")

# Selecionar top features por percentile de importância
importance_threshold = np.percentile(
    feature_imp_df['Importance'], 
    100 - CONFIG['lgb_importance_percentile']
)

features_final = feature_imp_df[
    feature_imp_df['Importance'] >= importance_threshold
]['Feature'].tolist()

print(f"    ✓ Features selecionadas: {len(features_final)}")
print(f"      - Threshold: {importance_threshold:.6f}")
print(f"      - Percentile: top {CONFIG['lgb_importance_percentile']}%")

# Visualizar cut-off
fig, ax = plt.subplots(figsize=(14, 8))
sorted_imp = feature_imp_df.sort_values('Importance', ascending=True)
colors = ['red' if x < importance_threshold else 'green' 
          for x in sorted_imp['Importance']]

ax.barh(range(len(sorted_imp)), sorted_imp['Importance'].values, color=colors, alpha=0.7)
ax.axvline(x=importance_threshold, color='black', linestyle='--', 
           label=f'Threshold = {importance_threshold:.6f}')
ax.set_yticks(range(min(50, len(sorted_imp))))
ax.set_yticklabels(sorted_imp['Feature'].values[-50:])
ax.set_xlabel("Feature Importance (Gain)")
ax.set_title("LightGBM Feature Importance - Threshold Cut-off")
ax.legend()
plt.tight_layout()
plt.savefig(f"./plots/lgb_importance_{timestamp}.png", dpi=300, bbox_inches='tight')
plt.show()

logger.info(f"LightGBM importance plot saved")

# ============================================================================
# 7. VALIDAÇÃO DO MODELO
# ============================================================================
print(f"\n[7] Validação do Modelo...")

# Predicações
y_pred_train = booster.predict(X_trn)
y_pred_val = booster.predict(X_val)

if objective == 'binary':
    # Para binary classification, converter probabilidades
    train_auc = roc_auc_score(y_trn, y_pred_train)
    val_auc = roc_auc_score(y_val, y_pred_val)
    
    print(f"    Train ROC-AUC: {train_auc:.4f}")
    print(f"    Valid ROC-AUC: {val_auc:.4f}")
    logger.info(f"ROC-AUC - Train: {train_auc:.4f}, Valid: {val_auc:.4f}")
else:
    train_loss = mean_squared_error(y_trn, y_pred_train)
    val_loss = mean_squared_error(y_val, y_pred_val)
    
    print(f"    Train MSE: {train_loss:.6f}")
    print(f"    Valid MSE: {val_loss:.6f}")
    logger.info(f"MSE - Train: {train_loss:.6f}, Valid: {val_loss:.6f}")

# ============================================================================
# 8. DISTRIBUIÇÃO DE IMPORTÂNCIA
# ============================================================================
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Histograma
ax[0].hist(feature_imp_df['Importance'], bins=50, edgecolor='black', alpha=0.7)
ax[0].axvline(importance_threshold, color='red', linestyle='--', 
              label=f'Threshold: {importance_threshold:.6f}')
ax[0].set_xlabel("Feature Importance")
ax[0].set_ylabel("Frequência")
ax[0].set_title("Distribuição de Feature Importance")
ax[0].legend()

# Gráfico de importância normalizada (top 30)
top_30 = feature_imp_df.head(30)
ax[1].barh(range(len(top_30)), top_30['Normalized_Importance'].values)
ax[1].set_yticks(range(len(top_30)))
ax[1].set_yticklabels(top_30['Feature'].values)
ax[1].set_xlabel("Importância Normalizada")
ax[1].set_title("Top 30 Features (Importância Normalizada)")
ax[1].invert_yaxis()

plt.tight_layout()
plt.savefig(f"./plots/lgb_importance_distribution_{timestamp}.png", dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 9. RESUMO FINAL
# ============================================================================
print(f"\n" + "="*80)
print(f"📊 RESUMO - LIGHTGBM FEATURE IMPORTANCE")
print(f"="*80)
print(f"\nFeatures de Entrada:      {len(X_train.columns)}")
print(f"Features Selecionadas:    {len(features_final)}")
print(f"Redução:                  {((len(X_train.columns) - len(features_final)) / len(X_train.columns) * 100):.1f}%")
print(f"\nÁrvores LightGBM:         {booster.num_trees()}")
if objective == 'binary':
    print(f"ROC-AUC (validação):      {val_auc:.4f}")
print(f"\n✓ LightGBM Feature Selection concluído")
print("="*80)

# ============================================================================
# 10. GUARDAR PARA PRÓXIMAS CÉLULAS
# ============================================================================
X_selected_lgb = X_train[features_final].copy()
booster_model = booster  # Guardar modelo para SHAP

logger.info(f"LightGBM feature selection complete: {len(features_final)} features")

# ============================================================================
# CÉLULA 5: SHAP EXPLAINABILITY
# ============================================================================

print("="*80)
print("EXPLAINABILITY - SHAP (SHapley Additive exPlanations)")
print("="*80)

# ============================================================================
# 1. CALCULAR VALORES SHAP COM TreeExplainer
# ============================================================================
print(f"\n[1] Calculando SHAP Values...")

# Usar TreeExplainer (otimizado para LightGBM)
# Usar sample de dados para explicação (para performance)
sample_size = min(5000, len(X_val))
sample_idx = np.random.choice(len(X_val), sample_size, replace=False)
X_sample = X_val.iloc[sample_idx].reset_index(drop=True)

print(f"    Usando sample de {len(X_sample)} amostras (de {len(X_val)})")

# Criar explainer SHAP
explainer = shap.TreeExplainer(booster_model)
logger.info(f"SHAP TreeExplainer criado")

# Calcular SHAP values
print(f"    ⏳ Calculando SHAP values (pode levar alguns minutos)...")
shap_values = explainer.shap_values(X_sample)

# Para classificação binária, pode retornar array ou lista
if isinstance(shap_values, list):
    # Usar SHAP values para classe positiva (default)
    shap_values = shap_values[1]
    print(f"    ✓ SHAP values calculados (classe positiva)")
else:
    print(f"    ✓ SHAP values calculados")

logger.info(f"SHAP values shape: {shap_values.shape}")

# ============================================================================
# 2. EXPLICAÇÃO DE VALORES BASE
# ============================================================================
print(f"\n[2] Interpretando valores base...")

base_value = explainer.expected_value
if isinstance(base_value, list):
    base_value = base_value[1]  # Classe positiva (default)

print(f"    Base Value (média de predições): {base_value:.6f}")
print(f"    Interpretação: Em média, o modelo prediz ~{base_value:.1%}")
logger.info(f"Base value: {base_value:.6f}")

# ============================================================================
# 3. RESUMO ESTATÍSTICO DE SHAP VALUES
# ============================================================================
print(f"\n[3] Estatísticas dos SHAP Values...")

shap_abs = np.abs(shap_values)
shap_mean = np.mean(shap_abs, axis=0)
shap_std = np.std(shap_abs, axis=0)

shap_stats = pd.DataFrame({
    'Feature': X_sample.columns,
    'Mean_Abs_SHAP': shap_mean,
    'Std_SHAP': shap_std,
    'Mean_Value': X_sample.mean(numeric_only=True).values,
    'Std_Value': X_sample.std(numeric_only=True).values,
})

shap_stats = shap_stats.sort_values('Mean_Abs_SHAP', ascending=False)

print(f"\n    Top 15 features por impacto SHAP:")
print(shap_stats.head(15)[['Feature', 'Mean_Abs_SHAP', 'Std_SHAP']].to_string(index=False))

# Salvar
shap_stats_path = f"./reports/shap_statistics_{timestamp}.csv"
shap_stats.to_csv(shap_stats_path, index=False)
logger.info(f"SHAP statistics saved to {shap_stats_path}")

# ============================================================================
# 4. SUMMARY PLOT (BEESWARM) - TOP FEATURES
# ============================================================================
print(f"\n[4] Gerando Summary Plot (Beeswarm)...")

fig, ax = plt.subplots(figsize=(12, 8))

# Usar pyplot para summary plot
shap.summary_plot(shap_values, X_sample, plot_type="beeswarm", 
                  max_display=15, show=False)

plt.title("SHAP Summary Plot - Impacto das Features nas Predições", 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel("SHAP Value (impacto na predição)", fontsize=11)
plt.tight_layout()
plt.savefig(f"./plots/shap_summary_beeswarm_{timestamp}.png", dpi=300, bbox_inches='tight')
plt.show()

logger.info(f"SHAP summary plot (beeswarm) saved")

# ============================================================================
# 5. SUMMARY BAR PLOT
# ============================================================================
print(f"\n[5] Gerando Summary Bar Plot...")

fig, ax = plt.subplots(figsize=(12, 8))

shap.summary_plot(shap_values, X_sample, plot_type="bar", 
                  max_display=15, show=False)

plt.title("SHAP Feature Importance - Impacto Médio Absoluto", 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel("Impacto Médio |SHAP value|", fontsize=11)
plt.tight_layout()
plt.savefig(f"./plots/shap_summary_bar_{timestamp}.png", dpi=300, bbox_inches='tight')
plt.show()

logger.info(f"SHAP summary plot (bar) saved")

# ============================================================================
# 6. DEPENDENCE PLOTS - TOP 5 FEATURES
# ============================================================================
print(f"\n[6] Gerando Dependence Plots (top 5 features)...")

top_n_dependence = min(5, len(X_sample.columns))
top_features_dependence = shap_stats.head(top_n_dependence)['Feature'].tolist()

fig, axes = plt.subplots(top_n_dependence, 1, figsize=(12, 4*top_n_dependence))

if top_n_dependence == 1:
    axes = [axes]

for idx, feature in enumerate(top_features_dependence):
    ax = axes[idx]
    
    shap.dependence_plot(feature, shap_values, X_sample, 
                        ax=ax, show=False)
    ax.set_title(f"Dependência SHAP: {feature}", fontweight='bold')

plt.tight_layout()
plt.savefig(f"./plots/shap_dependence_plots_{timestamp}.png", dpi=300, bbox_inches='tight')
plt.show()

logger.info(f"SHAP dependence plots saved")

# ============================================================================
# 7. FORCE PLOT PARA AMOSTRA INDIVIDUAL
# ============================================================================
print(f"\n[7] Gerando Force Plot (exemplo de predição individual)...")

# Selecionar amostras extremas (alto risk e baixo risk)
predictions = booster_model.predict(X_sample)
high_risk_idx = np.argmax(predictions)
low_risk_idx = np.argmin(predictions)

# Force plot para alto risco
try:
    fig = shap.force_plot(
        base_value, 
        shap_values[high_risk_idx:high_risk_idx+1], 
        X_sample.iloc[high_risk_idx:high_risk_idx+1],
        matplotlib=True,
        show=False
    )
    plt.title("Force Plot - Predição com ALTO RISCO de Default", 
              fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(f"./plots/shap_force_high_risk_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()
    logger.info(f"Force plot (high risk) saved")
except Exception as e:
    logger.warning(f"Force plot geração: {e}")

# ============================================================================
# 8. INTERPRETAÇÃO AUTOMÁTICA DOS RESULTADOS
# ============================================================================
print(f"\n" + "="*80)
print(f"📊 INTERPRETAÇÃO - INSIGHTS DE EXPLAINABILITY")
print(f"="*80)

print(f"\n🔍 CONCLUSÕES PRINCIPAIS:")

# Análise de direção dos efeitos
print(f"\n[1] Direção dos Efeitos:")

# Features que AUMENTAM risco (correlação positiva entre valor e SHAP)
corr_positive = []
for feature in X_sample.columns[:10]:  # Top 10 features
    feature_idx = X_sample.columns.get_loc(feature)
    corr = np.corrcoef(X_sample[feature].values, shap_values[:, feature_idx])[0, 1]
    
    if np.isfinite(corr):
        if corr > 0.2:
            corr_positive.append((feature, corr))

if corr_positive:
    corr_positive.sort(key=lambda x: x[1], reverse=True)
    print(f"\n    ⬆️ AUMENTAM risco de default (valores altos → maior default):")
    for feat, corr in corr_positive[:5]:
        print(f"       • {feat} (correlação SHAP: {corr:.3f})")

# Features que REDUZEM risco
corr_negative = []
for feature in X_sample.columns[:10]:
    feature_idx = X_sample.columns.get_loc(feature)
    corr = np.corrcoef(X_sample[feature].values, shap_values[:, feature_idx])[0, 1]
    
    if np.isfinite(corr):
        if corr < -0.2:
            corr_negative.append((feature, corr))

if corr_negative:
    corr_negative.sort(key=lambda x: x[1])
    print(f"\n    ⬇️ REDUZEM risco de default (valores altos → menor default):")
    for feat, corr in corr_negative[:5]:
        print(f"       • {feat} (correlação SHAP: {corr:.3f})")

print(f"\n[2] Top 3 Features Mais Impactantes:")
for idx, row in shap_stats.head(3).iterrows():
    print(f"    {idx+1}. {row['Feature']:30s} | Impacto: {row['Mean_Abs_SHAP']:8.6f}")

print(f"\n[3] Recomendações para Negócio:")
print(f"    • Focar em monitoramento das top 3 features")
print(f"    • Features com SHAP negativo = \"proteção\" contra default")
print(f"    • Solicitar informações adicionais se features de risco estiverem altas")

# ============================================================================
# 9. RESUMO FINAL DO PIPELINE
# ============================================================================
print(f"\n" + "="*80)
print(f"🎯 RESUMO FINAL - PIPELINE DE FEATURE SELECTION")
print(f"="*80)

print(f"\n✅ Pipeline Concluído com Sucesso!")
print(f"\n📊 REDUÇÃO DE FEATURES:")
print(f"   Iniciais:                  {len(df.columns) - 1} (excluindo target)")
print(f"   Após pré-processamento:    {len(X_clean.columns)}")
print(f"   Após Correlação + MI:      {len(features_selected_corr_mi)}")
print(f"   Após LightGBM Importance:  {len(features_final)}")
print(f"   Redução Total:             {((len(df.columns) - 1 - len(features_final)) / (len(df.columns) - 1) * 100):.1f}%")

print(f"\n📈 FEATURES FINAIS SELECIONADAS ({len(features_final)}):")
for idx, feat in enumerate(features_final[:20], 1):
    importance = feature_imp_df[feature_imp_df['Feature'] == feat]['Importance'].values
    if len(importance) > 0:
        print(f"   {idx:2d}. {feat:30s} | Importance: {importance[0]:10.6f}")

if len(features_final) > 20:
    print(f"   ... e mais {len(features_final) - 20} features")

print(f"\n📁 Artefatos Gerados:")
print(f"   • Modelo LightGBM: booster_model")
print(f"   • Features finais: features_final")
print(f"   • X selecionado: X_selected_lgb")
print(f"   • SHAP explainer: explainer")
print(f"   • Plots: ./plots/")
print(f"   • Relatórios: ./reports/")
print(f"   • Logs: {log_file}")

print(f"\n{'='*80}")
print(f"✨ FIM DO PIPELINE - PRONTO PARA MODELAGEM")
print(f"{'='*80}")

# Salvar lista final de features
final_features_path = f"./reports/final_selected_features_{timestamp}.txt"
with open(final_features_path, 'w') as f:
    f.write(f"SELECTED FEATURES ({len(features_final)} total)\n")
    f.write("="*60 + "\n\n")
    for idx, feat in enumerate(features_final, 1):
        f.write(f"{idx}. {feat}\n")

logger.info(f"PIPELINE COMPLETE - {len(features_final)} features selected")
logger.info(f"Final features saved to {final_features_path}")