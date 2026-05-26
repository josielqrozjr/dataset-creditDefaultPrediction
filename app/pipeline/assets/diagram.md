graph TD
    %% Configuração de Estilos e Cores (Padrão Acadêmico/Profissional)
    classDef raw fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef proc fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef split fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef final fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;

    %% Nodos do Pipeline
    A[<b>1. Dados Brutos (Série Temporal)</b><br>• train/data_*.parquet<br>• Volumetria: 5.531.451 linhas<br>• Variáveis: ~190 colunas]:::raw
    
    A --> B[<b>2. Engenharia Temporal (DuckDB)</b><br>• Janela: LAG OVER Partition por Cliente<br>• Criação de tendências de primeira ordem<br>• Métricas: _diff1 (Num) e _changed (Cat)]:::proc
    
    B --> C[<b>3. Agregação de Clientes (Polars)</b><br>• Operação: Group By customer_ID<br>• Redução de linhas e expansão de colunas<br>• Estrutura: 458.913 linhas × 3.264 colunas]:::proc
    
    C --> D[<b>4. Merge com Labels (Polars)</b><br>• Operação: Inner Join com train_labels<br>• Alinhamento do gabarito de default<br>• Estrutura: 458.913 linhas × 3.265 colunas <i>(+1 target)</i>]:::proc

    D --> E[<b>5. Split Estratificado 80/20 (Polars)</b><br>• Divisão por janelas de Ranks embaralhados<br>• Preservação da proporção original da classe]:::split

    E --> E1[<b>Conjunto de Validação/Teste (20%)</b><br>• Dimensões: 91.783 linhas × 3.265 colunas<br>• Proporção Target: 25.8937%<br>• <i>Isolado para avaliação final</i>]:::split
    
    E --> E2[<b>Conjunto de Treino Inicial (80%)</b><br>• Dimensões: 367.130 linhas × 3.265 colunas<br>• Proporção Target: 25.8933%]:::split

    E2 --> F[<b>6. Feature Selection (Somente no Treino)</b><br>• Algoritmo: Funil Permissivo + LightGBM Gain Importance<br>• Tratamento Nativo: 22 colunas categóricas mapeadas<br>• Balanceamento Algorítmico: is_unbalance=True]:::final

    F --> G[<b>Dataset de Treino Final Selecionado</b><br>• Estrutura: 367.130 linhas × 400 colunas<br>• Redução de ~88% das colunas irrelevantes<br>• <i>Pronto para a Modelagem Definitiva</i>]:::final