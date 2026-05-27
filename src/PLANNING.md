# Roadmap de Desenvolvimento: Benchmark AMEX

Este documento é o guia interno de engenharia para a construção do orquestrador de testes dos classificadores. 

## Fase 0: Infraestrutura e Configurações Base
- [X] Criar `src/evaluation/amex_metric.py` (Implementar a fórmula matemática oficial da AMEX: Gini + Top 4%).
- [ ] Atualizar `src/evaluation/metrics.py` (Adicionar suporte a `StratifiedKFold`).
- [ ] Atualizar `src/config.py` (Definir caminhos, *seeds*, dicionário de modelos atualizado, remover modelos descartados, configurar grades do Optuna).
- [ ] Atualizar arquivos base dos modelos em `src/models/` (Adicionar CatBoost, LightGBM, ajustar pipelines de Stacking/Blending).

## Fase 1: Provas de Conceito (Validação Metodológica)
- [ ] Desenvolver `run_phase1_poc.py`.
- [ ] Executar LR e XGBoost: Base Completa (3.265 features) *vs.* Base Enxuta (400 features).
- [ ] Executar LR e XGBoost: Sem Balanceamento *vs.* Undersampling *vs.* Scale Pos Weight / Class Weight.
- [ ] Exportar tabelas de resultados que justificarão as próximas fases.

## Fase 2: Campeonato Aberto (Baseline dos 10 Modelos)
- [ ] Desenvolver `run_phase2_benchmark.py`.
- [ ] Carregar base de treino enxuta (400 features), aplicar balanceamento algorítmico global.
- [ ] Treinar os 7 modelos individuais (LR, KNN, ANN, RF, XGB, LGBM, CatBoost) com hiperparâmetros default + `StratifiedKFold`.
- [ ] Coletar métricas OOF (Out-Of-Fold) e gerar ranking preliminar pelo AMEX Score.

## Fase 3: Otimização Suprema (Optuna + GPU)
- [ ] Desenvolver `run_phase3_optuna.py`.
- [ ] Isolar automaticamente o "Top 3" modelos da Fase 2 (provavelmente LGBM, XGB e CatBoost).
- [ ] Configurar função objetivo do Optuna maximizando a *AMEX Metric*.
- [ ] Executar otimização bayesiana (ex: 50 a 100 *trials*) utilizando aceleração por GPU.
- [ ] Salvar os hiperparâmetros campeões em disco (`best_params.json`).

## Fase 4: Meta-Classificadores (O Limite de Performance)
- [ ] Desenvolver `run_phase4_ensembles.py`.
- [ ] Construir *Voting Classifier* (Soft Voting) com os modelos otimizados da Fase 3.
- [ ] Construir *Stacking Classifier* (Garantindo ausência de leakage com CV OOF).
- [ ] Construir *Blending Classifier* (Com holdout explícito).
- [ ] Coletar métricas finais e coroar o modelo campeão absoluto.

## Fase 5: Relatórios e Visualização
- [ ] Desenvolver `src/evaluation/visualization.py` (Atualizado).
- [ ] Gerar gráficos de barras agrupadas (ROC-AUC *vs* AMEX Score).
- [ ] Gerar curvas Precision-Recall comparativas.
- [ ] Compilar CSV final unificando Treino (CV) e Validação/Teste (20% isolado).