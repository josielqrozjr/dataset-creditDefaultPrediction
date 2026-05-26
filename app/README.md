# Arquitetura Estrutural do Pipeline de Dados e Engenharia de Atributos

**Título do Trabalho:** Benchmarking de Algoritmos na Detecção de Inadimplência em Cartões de Crédito: De Modelos Tradicionais a Meta-Classificadores  
**Orientador:** Prof. Joelton  

Este documento apresenta a especificação técnica detalhada de todas as etapas que compõem o pipeline híbrido de processamento de dados desenvolvidos para a base da American Express (AMEX). O pipeline foi projetado sob a ótica de engenharia de software de alta performance, utilizando uma arquitetura combinada (**DuckDB + Polars**) para viabilizar o processamento de mais de 5.5 milhões de registros sem estouro de memória RAM (*Out of Memory*), preparando a base de dados para o subsequente benchmark de 10 modelos de Machine Learning.

---

## 1. Fase Inicial: Carga de Dados e Extração de Schema
Antes do início de qualquer transformação matemática, o pipeline executa um mapeamento dinâmico de tipos através do DuckDB diretamente sobre arquivos Parquet particionados nativos em disco (`data_*.parquet`). O dataset bruto possui um formato longitudinal (série temporal incompleta) composto por **5.531.451 linhas** distribuídas por faturas mensais de clientes, contendo cerca de 190 variáveis originais altamente mascaradas e normalizadas.

---

## 2. Fase 1: Engenharia Temporal de Primeira Ordem (DuckDB)
Esta etapa atua na dimensão longitudinal dos dados. O objetivo é capturar a dinâmica de evolução financeira do cliente de uma fatura para a próxima, gerando atributos baseados no histórico sequencial imediato. A computação é otimizada via *Window Functions* SQL executadas diretamente em disco pelo DuckDB.

### Tabela 1: Atributos de Tendência Sequencial (Fase Temporal)

| Atributo / Sufixo | Tipo de Dado | Descrição Matemática / Lógica | Objetivo Preditivo no Cenário de Risco |
| :--- | :--- | :--- | :--- |
| `[Feature]_diff1` | Numérico (`Float32`) | $X_{t} - X_{t-1}$ <br>Onde $X_t$ é o valor da variável na fatura atual e $X_{t-1}$ é o valor na fatura imediatamente anterior, calculados através da janela `LAG() OVER (PARTITION BY customer_ID ORDER BY S_2)`. | Medir a volatilidade, aceleração de gastos ou velocidade de endividamento de curto prazo do tomador de crédito. |
| `[Feature]_changed`| Binário (`Int8`: 0 ou 1) | $\begin{cases} 1, & \text{se } X_{t} \neq X_{t-1} \\ 0, & \text{se } X_{t} = X_{t-1} \end{cases}$ <br>Retorna `0` se o primeiro registro for nulo (fatura inicial). | Mapear transições explícitas de estado comportamental ou cadastral ocorridas entre faturas consecutivas. |

### O Papel das Variáveis Categóricas na Engenharia Temporal
A base da AMEX possui 11 variáveis categóricas mapeadas oficiais (`B_30`, `B_38`, `D_114`, `D_116`, `D_117`, `D_120`, `D_126`, `D_63`, `D_64`, `D_66`, `D_68`). 
Nesta etapa, elas recebem um tratamento de **bloqueio de colinearidade aritmética**:
* **Isolamento de Redundância:** É matematicamente inválido calcular a diferença numérica (`_diff1`) sobre dados categóricos, mesmo quando mascarados como inteiros (ex: subversão de status `3` menos status `1` não possui significado econômico ou de crédito). 
* **Extração de Instabilidade:** Em contrapartida, as variáveis categóricas são as geradoras exclusivas das flags `_changed`. Identificar se um cliente alterou seu código de comportamento de bureaus ou tipo de residência financeira (`_changed = 1`) indica quebras de padrão de estabilidade, um forte sinal preditivo para a iminência de um *default*.

---

## 3. Fase 2: Agregação Latitudinal de Clientes (Polars)
Esta etapa executa uma transformação geométrica na matriz de dados, convertendo o formato longitudinal (múltiplas faturas por cliente) para um formato estritamente latitudinal (**uma única linha por cliente**). O motor *Lazy* escrito em Rust/C++ do Polars condensa os 5.531.451 registros brutos em exatamente **458.913 clientes únicos**, gerando uma matriz robusta de **3.264 colunas** de alta densidade informativa.

### Tabela 2: Métricas de Compressão e Perfis Estatísticos (Agregação)

| Atributo Gerado / Sufixo | Escopo de Aplicação | Operação Matemática Base | Racional de Risco de Crédito |
| :--- | :--- | :--- | :--- |
| `_mean` | Variáveis Numéricas | $\frac{1}{n}\sum_{i=1}^{n} X_i$ | Fornece o patamar médio histórico de gastos, saldos ou atrasos do cliente, removendo ruídos sazonais isolados. |
| `_std` | Variáveis Numéricas | $\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(X_i - \bar{X})^2}$ | Quantifica a instabilidade financeira. Altos desvios indicam descontrole no fluxo de caixa ou picos atípicos de utilização de limite. |
| `_min` | Variáveis Numéricas | Valor Mínimo Retido | Identifica a menor exposição ou melhor estado histórico de liquidez financeira do cliente na série estudada. |
| `_max` | Variáveis Numéricas | Valor Máximo Retido | Captura o teto de estresse financeiro, pico de endividamento ou maior atraso registrado em bureau. |
| `_last` | Todas as Variáveis | Último valor temporal disponível | O estado mais recente e atualizado do cliente antes do fechamento da janela de observação. Possui peso crítico para o modelo. |
| `_first` | Categóricas Originais | Primeiro valor temporal registrado | Registra o ponto de partida do cliente na série histórica, servindo de âncora comportamental para análises de evolução. |
| `_total_delta` | Numéricas Originais | $X_{last} - X_{first}$ | Medir a variação absoluta líquida do cliente entre o início e o fim do período monitorado (crescimento ou redução de exposição). |
| `_trend_ratio` | Numéricas Originais | $\frac{X_{last}}{\bar{X} + 1e-5}$ | Fornece uma métrica de desvio recente. Identifica se o comportamento atual está acima ou abaixo da média histórica do próprio indivíduo. |
| `_pos_ratio` | Numéricas Originais | $\frac{1}{n}\sum \mathbb{I}(Diff_t > 0)$ | Proporção de meses em que o cliente aumentou o valor da variável. Indica tendência de crescimento persistente da métrica. |
| `_avg_monthly_slope` | Numéricas Originais | $\frac{X_{last} - X_{first}}{\text{Count de Meses}}$ | Proxy simplificada de inclinação linear. Avalia o ritmo ou velocidade média mensal com que a feature financeira variou. |
| `_total_changes` | Flags `_changed` | $\sum (X_{changed})$ | Soma total das ocorrências de transições. Mede o volume bruto de instabilidade ou mudanças de postura do cliente. |
| `_change_frequency` | Flags `_changed` | Média das flags `_changed` | Percentual de meses em que o cliente sofreu alteração de categoria, indicando volatilidade sistêmica ou comportamental. |
| `_changed_recently` | Flags `_changed` | Último estado da flag | Indica se o cliente alterou de comportamento cadastral ou de perfil exatamente na fatura mais recente. |
| `_nunique` | Categóricas Originais | Contagem de valores distintos | Avalia a diversidade de estados. Clientes que transitam por muitas categorias diferentes de bureaus exibem perfis erráticos. |
| `_count` | Todas as Variáveis | Contagem de registros válidos | Mede a densidade da informação. Em risco, a quantidade de meses preenchidos reflete a maturidade da conta e o tempo de relacionamento. |

### O Papel das Variáveis Categóricas na Agregação
Nesta fase, ocorre a separação de rotas algorítmicas estruturais no Polars para evitar contaminação estatística:
* **Bloqueio de Métricas Contínuas:** As 11 variáveis categóricas originais têm o cálculo de `mean`, `std`, `min` e `max` rigidamente bloqueados. Calcular a média aritmética de categorias mascaradas induziria os futuros modelos de benchmark a assumirem uma ordenação linear inexistente.
* **Agregação Baseada em Estado e Frequência:** O comportamento das categóricas é resumido puramente por métricas de estado atual (`_last`), estado histórico inicial (`_first`), contagem de transições distintas (`_nunique`) e atividade (`_count`). Além disso, as flags binárias de mudança geradas no DuckDB (`_changed`) entram em uma rota matemática secundária exclusiva, onde são sumarizadas via contagem absoluta (`_total_changes`) e taxa de ocorrência cumulativa (`_change_frequency`).

---

## 4. Fase 3: Fusão de Rótulos (*Merge*) e Amostragem Estratificada
Para garantir a integridade estatística do projeto e impedir vazamento de dados (*data leak*), o pipeline executa o cruzamento (*Inner Join*) com a base de gabarito e divide os dados utilizando o motor de janela do Polars de forma puramente nativa em memória RAM.

### Tabela 3: Volumetria e Distribuição de Classes pós-Estratificação

| Subconjunto Gerado | Volumetria (Linhas) | Proporção de Colunas | Distribuição da Classe Alvo (`target = 1`) | Função Arquitetural no TCC |
| :--- | :--- | :--- | :--- | :--- |
| **Dataset Total Consolidado** | 458.913 | 3.265 *(3.264 + target)* | ~25.8936% de Inadimplência | Base consolidada em memória após o Join relacional. |
| **Treino Inicial (80%)** | **367.130** | 3.265 | **25.8933%** | Submetido exclusivamente à seleção de atributos e treinamento algorítmico. |
| **Validação / Teste (20%)** | **91.783** | 3.265 | **25.8937%** | Base mantida sob isolamento absoluto (gabarito oculto) para simular o ambiente produtivo real. |

*Nota Metodológica:* A variação infinitesimal na proporção do target observada entre Treino e Teste (apenas $0.0004\%$) valida cientificamente a precisão da técnica de divisão baseada em *funções de ordenação por janela* do Polars, assegurando que ambos os conjuntos sejam réplicas populacionais idênticas.

---

## 5. Fase 4: Seleção Híbrida de Atributos Contextualizada (Polars + LightGBM)
Visando contornar a **Maldição da Dimensionalidade** (reduzindo de 3.265 para **400 colunas** de alto impacto), esta etapa aplica um funil de filtragem rigoroso implementado estritamente sobre a base de treino de 80%. Esta abordagem descarta técnicas tradicionais de balanceamento físico (como SMOTE ou Undersampling), adotando em seu lugar o **Aprendizado Sensível ao Custo** (*Cost-Sensitive Learning*).

### Tabela 4: Especificação do Funil de Feature Selection Inteligente

| Filtro / Estágio | Tipo de Filtro | Threshold Aplicado | Racional Técnico e Contexto AMEX |
| :--- | :--- | :--- | :--- |
| **Filtro de Missing Absoluto** | Estático (Polars) | `0.999` (99.9%) | **Preservação de Dados Ausentes (MNAR):** Remove apenas colunas 100% vazias. Variáveis com alta taxa de nulos (ex: 95%) são mantidas intencionalmente, pois a ausência do dado em crédito é um forte sinal preditivo que o LightGBM converte em regras automáticas de decisão. |
| **Filtro de Quase-Constantes** | Estático (Polars) | `0.999` (99.9%) | Descarta variáveis sem variância estatística, onde o mesmo valor se repete de forma unânime na base, economizando poder de processamento. |
| **Filtro de Colinearidade Exata**| Estático (Pandas) | `0.98` (98.0%) | Identifica e remove "recursos clones" altamente correlacionados gerados pelas agregações matemáticas, eliminando redundância severa. |
| **Filtro de Importância por Árvore**| Algorítmico (LightGBM)| Top **400** Melhores Features baseadas em **Gain** | Treinamento de um estimador auxiliar veloz de 200 iterações. O cálculo do ganho avalia o quanto cada feature efetivamente reduz a entropia (entropia de Shannon) da inadimplência. Recursos com ganho zero ou irrelevantes são eliminados. |
| **Balanceamento Algorítmico** | Hiperparâmetro | `is_unbalance: True` | Atuando dentro do LightGBM auxiliar, ele altera os pesos na função de perda (*Loss Function*). Isso força a árvore avaliadora a penalizar severamente o erro na classe minoritária, garantindo que colunas cruciais para prever inadimplentes não fossem descartadas em prol da maioria. |

---

## 6. Próximos Passos: Execução do Benchmarking de Classificadores
Com a conclusão bem-sucedida deste pipeline, os dados encontram-se matematicamente higienizados, enriquecidos e dimensionalmente otimizados. Conforme o escopo delimitado pelo título escolhido, a matriz de treino resultante de **400 colunas selecionadas** e a matriz de validação isolada servirão como base para a execução de um amplo e rigoroso **Benchmarking de 10 Modelos**, cobrindo o espectro completo da evolução da ciência de dados:

1. **Modelos Tradicionais (Linha de Base):** *Logistic Regression, KNN (K-Nearest Neighbors)* e *Decision Tree*.
2. **Modelos Baseados em Boosting e Ensembles Homogêneos:** *AdaBoost, Random Forest* e *XGBoost*.
3. **Arquiteturas de Redes Neurais:** *ANN (Multi-Layer Perceptron - MLP)*.
4. **Meta-Classificadores Avançados (Ensembles Heterogêneos):** *Voting Classifier, Blending* (com Logistic Regression, KNN, Random Forest, XGBoost) e *Stacking* (com Naive Bayes, KNN, Logistic Regression).

Todos os modelos serão avaliados competitivamente sob as métricas de ROC-AUC, F1-Score, Precisão e a métrica de sensibilidade oficial estipulada pela American Express.
