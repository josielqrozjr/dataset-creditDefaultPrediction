# Diretório de Modelos e Benchmarking: Detecção de Inadimplência

Este diretório contém os códigos e as definições metodológicas para a avaliação dos algoritmos de classificação. O objetivo desta etapa é comparar o desempenho de diferentes modelos matemáticos na previsão de inadimplência (target) utilizando a base de dados tratada nas fases anteriores.

Para evitar alto custo computacional e garantir a validade científica, a avaliação foi dividida em um funil de quatro fases. 

---

## 1. Estrutura de Fases do Benchmark

A execução dos modelos segue uma ordem de validação por etapas. A saída de uma fase justifica as configurações da fase seguinte.

| Fase | Objetivo | Escopo de Execução | Justificativa Metodológica |
| :--- | :--- | :--- | :--- |
| **Fase 1: Provas de Conceito** | Validar empiricamente a utilidade do Feature Selection (FS) e a abordagem de Balanceamento. | Modelos: Regressão Logística e XGBoost.<br>Testes: (Com FS vs. Sem FS) e (Algorítmico vs. Undersampling vs. Sem Balanceamento). | Evita testar todas as combinações em todos os modelos. Comprova as hipóteses iniciais nos dois extremos algorítmicos (um modelo linear e um ensemble de árvores). |
| **Fase 2: Campeonato Aberto** | Estabelecer a linha de base de desempenho para todos os algoritmos escolhidos. | 10 Modelos rodando com hiperparâmetros padrão (default), base enxuta (400 variáveis) e balanceamento algorítmico. | Permite uma comparação nivelada e com baixo custo computacional inicial. Identifica os algoritmos que melhor se adaptam à estrutura de dados da AMEX. |
| **Fase 3: Otimização de Hiperparâmetros** | Encontrar o limite de performance dos melhores algoritmos identificados na Fase 2. | Algoritmo Optuna aplicado apenas no Top 3 modelos da Fase 2. Uso de aceleração por GPU. | O método de otimização bayesiana (Optuna) é mais rápido e eficiente que o GridSearch tradicional. Aplicá-lo apenas no Top 3 economiza recursos de processamento. |
| **Fase 4: Meta-Classificadores** | Combinar as previsões dos modelos otimizados para reduzir variância e erro. | Stacking, Blending e Voting Classifier utilizando os modelos ajustados na Fase 3. | Modelos baseados na combinação de classificadores tendem a apresentar maior robustez contra *overfitting*. |

---

## 2. Seleção e Justificativa dos Algoritmos

O benchmark final é composto por 10 algoritmos, distribuídos em quatro categorias principais. Modelos clássicos como *Decision Tree* e *AdaBoost* foram removidos do escopo. A *Decision Tree* sofre *overfitting* em dados de alta dimensão, enquanto o *AdaBoost* é sensível a *outliers* (comuns em dados financeiros). 

Para substituí-los, foram incluídos o CatBoost e o LightGBM. A tabela abaixo detalha a escolha do portfólio de teste:

| Categoria | Modelo | Justificativa Técnica para Inclusão |
| :--- | :--- | :--- |
| **Modelos Tradicionais (Baseline)** | **Regressão Logística (LR)** | Representante linear paramétrico. Sensível à multicolinearidade. Usado para provar a eficácia da seleção de variáveis. |
| | **K-Nearest Neighbors (KNN)** | Modelo baseado em distância. Útil para capturar agrupamentos locais de perfis de inadimplência. |
| **Redes Neurais** | **ANN (Multi-Layer Perceptron)** | Capacidade de mapear interações não-lineares complexas através de múltiplas camadas ocultas. |
| **Ensembles Homogêneos (Árvores)** | **Random Forest** | Baseado em *Bagging*. Cria variabilidade usando amostras e subconjuntos de variáveis, reduzindo a variância geral em relação a uma árvore simples. |
| | **XGBoost** | Baseado em *Gradient Boosting*. Estrutura robusta contra valores ausentes e alto desempenho de convergência, com suporte nativo a GPU. |
| | **LightGBM** | Algoritmo rápido que constrói árvores por folha (leaf-wise). Incluído por ter sido o algoritmo que extraiu as métricas de importância no Feature Selection. |
| | **CatBoost** | Baseado em árvores simétricas. Lida de forma otimizada com as 22 variáveis categóricas mapeadas no pipeline sem a necessidade de codificação prévia. |
| **Meta-Classificadores** | **Voting Classifier** | Combina as previsões dos modelos base por média simples de probabilidade (*soft voting*), suavizando os erros individuais. |
| | **Stacking** | Treina um modelo final sobre as previsões (usando validação cruzada) dos modelos base. Corrige tendências de erro dos classificadores subjacentes. |
| | **Blending** | Variação do Stacking que utiliza uma partição *holdout* fixa para treinar o meta-classificador. Mais rápido computacionalmente e mitiga o risco de vazamento de dados. |

---

## 3. Diretrizes de Execução e Prevenção de Vazamento de Dados

Algumas decisões técnicas foram adotadas para garantir a integridade dos resultados:

* **Isolamento da Base de Teste:** A base de teste (20%) não é exposta a nenhuma etapa do *Feature Selection* ou balanceamento de dados. 
* **Espelhamento de Dimensão:** O filtro de seleção (400 colunas) é calculado apenas na base de treino. A lista resultante (arquivo `.txt`) é usada como máscara para aplicar o mesmo corte nas colunas da base de teste antes da entrada nos modelos. Isso garante conformidade de formato sem vazamento de informação (*data leakage*).
* **Foco no Balanceamento Algorítmico:** Devido à natureza financeira da base, técnicas de superamostragem (*oversampling*) física, como SMOTE, tendem a criar dados sintéticos irreais. O benchmark priorizará o balanceamento via função de custo (ex: `class_weight` ou `scale_pos_weight`), penalizando matematicamente os erros cometidos na classe minoritária.
* **Aceleração por Hardware:** Modelos compatíveis (como XGBoost e CatBoost) serão executados via placa de vídeo dedicada (GPU) durante a Fase 3 para viabilizar a otimização de dezenas de hiperparâmetros em tempo hábil.
Essa é uma excelente visão de organização de projeto. No mundo da engenharia de software e pesquisa reproduzível, **a melhor prática é separar os dois documentos**.
* **Validação Cruzada Estratificada:** Durante todas as fases de treinamento e otimização, os modelos são submetidos a um `StratifiedKFold` (5 partições). Isso garante que a proporção da classe minoritária (25,89%) seja matematicamente idêntica em cada dobra de validação, estabilizando as métricas e mitigando o risco de sobreajuste local.
* **Avaliação Orientada ao Domínio Financeiro:** Embora métricas padrão como ROC-AUC e F1-Score sejam coletadas, a otimização dos hiperparâmetros (Fase 3) buscará maximizar a **Métrica Oficial da AMEX** (uma composição ponderada entre o Índice de Gini Normalizado e a taxa de captura de *default* nos top 4% de risco). Isso alinha o modelo estritamente às necessidades de negócio de uma instituição financeira real.
* **Preservação Categórica Nativa:** Variáveis originalmente categóricas e indicadoras de transição de estado não foram submetidas a *One-Hot Encoding* prévio. Elas mantêm sua tipagem nominal na base selecionada, permitindo que algoritmos como CatBoost e LightGBM construam partições de árvore otimizadas internamente, preservando a densidade da informação.

---

## 4. Defesa Metodológica da Redução de Dimensionalidade

A base completa gerada na etapa de agregação possui 3.265 variáveis. No entanto, ela será utilizada integralmente apenas durante a **Fase 1**. A partir da Fase 2, o benchmark prosseguirá exclusivamente com as 400 variáveis mantidas pelo *Feature Selection*. Essa decisão estrutural baseia-se em três pilares metodológicos:

1. **Validação de Hipótese Empírica:** A Fase 1 atua como prova empírica. Ao comparar a performance da Regressão Logística e do XGBoost na base completa *vs.* base reduzida, o objetivo é confirmar que as 400 variáveis concentram a capacidade preditiva do conjunto, indicando que as 2.865 variáveis eliminadas representam colinearidade ou ruído.
Em ciência, nós não assumimos que o nosso *Feature Selection* é perfeito; nós provamos isso.
A sua hipótese é: *"As 400 colunas selecionadas pelo LightGBM contêm praticamente toda a informação útil para prever a inadimplência, e as outras 2.865 colunas são ruído ou redundância"*.

* A Fase 1 existe única e exclusivamente para **comprovar essa hipótese empiricamente**. Quando você rodar o XGBoost com as 3.265 colunas e depois com as 400 colunas, você observará que o ROC-AUC será quase idêntico (ou até melhor na base menor), mas o tempo de treino cairá drasticamente. Uma vez provado isso no papel (com tabelas e números no seu TCC), você não precisa mais carregar o "peso morto" de 2.800 colunas inúteis para o resto da pesquisa.

2. **Mitigação da Maldição da Dimensionalidade:** Executar os 10 algoritmos em mais de 3.000 dimensões causaria falhas algorítmicas pontuais. O cálculo de proximidade geométrica do KNN perde precisão matemática em alta dimensionalidade, enquanto a Regressão Logística e as Redes Neurais (ANN) enfrentariam multicolinearidade extrema, prejudicando a convergência. Avalie o impacto:
* **KNN:** Ele calcula distâncias geométricas entre os clientes. Em 3.000 dimensões, a matemática do KNN entra em colapso (as distâncias ficam todas iguais). Ele perderia a precisão completamente.
* **Regressão Logística e Redes Neurais (ANN):** Teriam que calcular pesos para 3.000 variáveis, a grande maioria sendo pura colinearidade (variáveis que dizem a mesma coisa). Isso causaria um *overfitting* violento e problemas de convergência matemática.

3. **Princípio da Parcimônia (Navalha de Ockham):** Sob a premissa estatística de que modelos mais simples devem ser priorizados quando entregam performance comparável, a manutenção sistêmica de 3.265 variáveis representaria um consumo injustificável de recursos computacionais (RAM e VRAM), lentificando as fases de otimização estendida. Na academia, existe um princípio de que *"entre dois modelos com a mesma performance, o modelo mais simples é sempre o melhor"*.

* Se a Fase 1 provar que 400 colunas entregam o mesmo poder de fogo que 3.000 colunas, manter as 3.000 para o resto do campeonato seria apenas um desperdício injustificável de energia elétrica e memória da GPU. O seu *Feature Selection* criou um novo "Padrão Ouro" de dados para a sua pesquisa.

## 5. Métricas de Avaliação do Benchmark

A avaliação dos classificadores utiliza um conjunto de métricas adequadas para bases com desbalanceamento de classes e aplicáveis ao domínio financeiro. A otimização não depende de uma métrica única, garantindo a análise sob diferentes perspectivas de custo e eficiência.

A tabela a seguir detalha o escopo de avaliação do benchmark:

| Métrica | Descrição e Cálculo | Justificativa no Contexto de Crédito |
| :--- | :--- | :--- |
| **Métrica Oficial AMEX** | Média entre o Gini Normalizado Ponderado e a Taxa de Captura nos Top 4% de risco. | Métrica principal (Norte) da Fase 3. Reflete a prioridade do negócio: ordenar os clientes por probabilidade de *default* e identificar os piores ofensores na faixa de maior risco. |
| **ROC-AUC** | Área sob a curva ROC (Taxa de Verdadeiros Positivos *vs.* Falsos Positivos). | Avalia a capacidade global de separação do modelo entre inadimplentes e bons pagadores, independentemente do limiar (threshold) de corte escolhido. |
| **AUPRC (Average Precision)** | Área sob a curva de Precision *vs.* Recall. | Métrica superior à ROC-AUC para cenários de alto desbalanceamento. Penaliza duramente modelos que geram muitos falsos positivos na classe minoritária. |
| **F1-Score** | Média harmônica matemática entre Precision e Recall. | Força o equilíbrio do modelo. Impede que algoritmos obtenham pontuações altas apenas aprovando todos os clientes ou apenas negando crédito para todos. |
| **Precision (Precisão)** | Verdadeiros Positivos / (Verdadeiros Positivos + Falsos Positivos). | Mede o custo do alarme falso. Baixa precisão indica que o modelo está negando crédito para bons pagadores, gerando perda de receita para a instituição. |
| **Recall (Sensibilidade)** | Verdadeiros Positivos / (Verdadeiros Positivos + Falsos Negativos). | Mede a capacidade de proteção financeira. Baixo recall indica que o modelo falhou em detectar inadimplentes reais, gerando prejuízo direto por calote. |

# 6. Roadmap Visual do Benchmark

```mermaid
graph TD
    A[Fase 1: Provas de Conceito<br/>Validação de FS e Balanceamento] -->|Justifica o corte para 400 variáveis| B(Fase 2: Campeonato Aberto<br/>10 Modelos com Configurações de Fábrica)
    B -->|Elimina os 7 piores| C(Fase 3: Otimização Optuna<br/>Busca Matemática Extrema no Top 3)
    C -->|Exporta os 3 Modelos Elite| D{Fase 4: Meta-Classificadores}
    D --> E[Voting Classifier<br/>Média dos 3]
    D --> F[Stacking Classifier<br/>Um modelo aprende com os 3]
    D --> G[Blending Classifier<br/>Holdout dos 3]
    E -.-> H((Comparação Final))
    F -.-> H((Comparação Final))
    G -.-> H((Comparação Final))
```