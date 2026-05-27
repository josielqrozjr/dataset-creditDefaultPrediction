"""
Métrica Oficial de Avaliação - American Express Default Prediction
------------------------------------------------------------------
Esta métrica é utilizada para ranquear os modelos no benchmark da AMEX.
Ela avalia a capacidade do modelo de ordenar corretamente os clientes 
pelo risco de inadimplência, atribuindo pesos diferentes para as classes.

A métrica M é a média aritmética entre duas sub-métricas:
1. Gini Normalizado Ponderado (Weighted Normalized Gini)
2. Taxa de Captura nos Top 4% (Top 4% Capture Rate)

Notas Técnicas:
- A classe negativa (target=0) recebe um peso de 20.
- A classe positiva (target=1) recebe um peso de 1.
- A implementação abaixo utiliza NumPy puro em vetorização para máxima 
  performance (cerca de 30x mais rápida que a versão em Pandas), sendo 
  essencial para não criar gargalos durante a otimização com o Optuna.
"""

import numpy as np

def amex_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula a Métrica Oficial da AMEX (Gini Normalizado + Recall @ 4%).
    
    Parâmetros:
    -----------
    y_true : np.ndarray ou list
        Array 1D com as classes reais (0 ou 1).
    y_pred : np.ndarray ou list
        Array 1D com as probabilidades preditas de inadimplência (target = 1).
        
    Retorna:
    --------
    float
        O valor da métrica AMEX (M), variando até o máximo de 1.0.
    """
    # Converter entradas para arrays flat do numpy garantindo formato consistente
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Cria uma matriz com a combinação de verdadeiros e preditos: [target, probabilidade]
    labels = np.transpose(np.array([y_true, y_pred]))
    
    # -------------------------------------------------------------
    # 1. Taxa de Captura nos Top 4% (Top 4% Capture Rate)
    # -------------------------------------------------------------
    # Ordenar as predições em ordem decrescente (do mais arriscado para o menos arriscado)
    labels_sorted_by_pred = labels[labels[:, 1].argsort()[::-1]]
    
    # Aplicar pesos estatísticos (20 para a classe majoritária, 1 para a minoritária)
    weights = np.where(labels_sorted_by_pred[:, 0] == 0, 20.0, 1.0)
    
    # Encontrar a linha de corte equivalente a 4% do peso total da base
    cutoff_weight = int(0.04 * np.sum(weights))
    
    # Filtrar apenas os registros até atingir a linha de corte de peso (Top 4%)
    cut_vals = labels_sorted_by_pred[np.cumsum(weights) <= cutoff_weight]
    
    # A captura é a soma de inadimplentes encontrados no top 4% dividida pelo total real
    top_four_capture = np.sum(cut_vals[:, 0]) / np.sum(labels_sorted_by_pred[:, 0])
    
    # -------------------------------------------------------------
    # 2. Gini Normalizado Ponderado (Weighted Normalized Gini)
    # -------------------------------------------------------------
    gini = [0.0, 0.0]
    
    # O loop calcula a curva com nosso modelo (i=1) e a curva de "Gabarito Perfeito" (i=0)
    for i in [1, 0]:
        labels_temp = np.transpose(np.array([y_true, y_pred]))
        
        # Ordenação do modelo vs Ordenação teórica perfeita
        if i == 1:
            labels_temp = labels_temp[labels_temp[:, 1].argsort()[::-1]]
        else:
            labels_temp = labels_temp[labels_temp[:, 0].argsort()[::-1]]
            
        weight = np.where(labels_temp[:, 0] == 0, 20.0, 1.0)
        
        # Eixo X teórico da Curva de Lorentz
        weight_random = np.cumsum(weight / np.sum(weight))
        
        # Eixo Y empírico da Curva de Lorentz
        total_pos = np.sum(labels_temp[:, 0] * weight)
        cum_pos_found = np.cumsum(labels_temp[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        
        # Integral do Gini
        gini[i] = np.sum((lorentz - weight_random) * weight)
        
    # Gini do modelo normalizado pelo teto matemático (Gini máximo perfeito)
    normalized_gini = gini[1] / gini[0]

    # -------------------------------------------------------------
    # 3. Cálculo Final Combinado
    # -------------------------------------------------------------
    return 0.5 * (normalized_gini + top_four_capture)


# =====================================================================
# Wrappers Nativos (Ponteiros para otimizadores nativos de Boosting)
# =====================================================================

def xgb_amex_metric(y_pred: np.ndarray, dmatrix) -> tuple[str, float]:
    """Wrapper da métrica para o formato exigido pelo XGBoost (XGBClassifier)."""
    y_true = dmatrix.get_label()
    return 'amex_score', amex_metric(y_true, y_pred)


def lgb_amex_metric(y_pred: np.ndarray, dataset) -> tuple[str, float, bool]:
    """Wrapper da métrica para o formato exigido pelo LightGBM (LGBMClassifier)."""
    y_true = dataset.get_label()
    # O retorno exige a flag 'is_higher_better = True' no final
    return 'amex_score', amex_metric(y_true, y_pred), True