# Base de Dados para Predição de Inadimplência

## 1. Apresentação

Este repositório documenta uma base de dados voltada ao problema de **predição de inadimplência** a partir da competição pública **American Express - Default Prediction** (Kaggle):

- Competição: <https://www.kaggle.com/competitions/amex-default-prediction/overview>

O objetivo central é organizar uma estrutura simples e reproduzível para preparação dos dados, com ênfase na conversão de arquivos CSV para o formato Parquet, visando maior eficiência de armazenamento e leitura.

Este projeto tem escopo exclusivo de **disponibilização e padronização da base** para consumo por outros repositórios.

## 2. Contexto do Problema

A tarefa proposta na competição consiste em estimar a probabilidade de um cliente entrar em estado de inadimplência no período de observação definido pelo desafio. Trata-se de um problema supervisionado de classificação binária, no qual:

- a classe positiva representa clientes com default;
- a classe negativa representa clientes sem default.

Em aplicações financeiras, esse tipo de modelagem é relevante para gestão de risco de crédito, definição de políticas de concessão e monitoramento da carteira.

## 3. Objetivo deste Repositório

Este projeto está orientado para:

- padronizar a organização local dos dados brutos;
- converter grandes arquivos CSV em Parquet com compressão eficiente;
- disponibilizar artefatos de dados prontos para serem consumidos por outros repositórios.

### 3.1 Escopo e Fronteira

Este repositório:

- **inclui** ingestão de arquivos CSV, conversão para Parquet e organização da estrutura de dados;
- **não inclui** experimentação, engenharia de atributos avançada, treinamento, validação ou publicação de modelos.

Essas etapas devem ocorrer em repositórios consumidores, de acordo com os objetivos de cada trabalho.

## 4. Estrutura do Projeto

```text
dataset-creditDefaultPrediction/
├── convertParquet.py
├── requirements.txt
├── README.md
└── data/
    ├── raw/                  # entrada esperada para CSVs do Kaggle
    └── parquet/
        ├── train/            # saída parquet do conjunto de treino
        └── test/             # saída parquet do conjunto de teste
```

Observação: as pastas podem ser criadas automaticamente pelo script quando necessário.

## 5. Dados da Competição

De forma geral, a competição disponibiliza arquivos para treino, teste e submissão. Neste repositório, o script está configurado para processar especificamente:

- `train_data.csv`;
- `test_data.csv`.

Esses arquivos devem ser posicionados em `data/raw/` antes da execução.

### 5.1 Observações importantes

- Os dados da competição não acompanham este repositório por questões de licenciamento e tamanho.
- O uso da base deve respeitar os termos da plataforma Kaggle e as regras da competição.

## 6. Preparação do Ambiente

### 6.1 Requisitos

- Python 3.10 ou superior (recomendado);
- Dependência principal: `duckdb`.

### 6.2 Instalação

```bash
pip install -r requirements.txt
```

## 7. Conversão de CSV para Parquet

O arquivo `convertParquet.py` realiza a conversão dos dados brutos para Parquet utilizando DuckDB em memória.

### 7.1 Estratégia adotada no script

- leitura dos CSVs com `read_csv_auto`;
- tolerância a erros de leitura (`ignore_errors=true`);
- inferência de tipos com varredura completa (`sample_size=-1`);
- compressão `ZSTD`;
- escrita paralelizada (`PER_THREAD_OUTPUT`);
- definição de `ROW_GROUP_SIZE` para controle de desempenho.

### 7.2 Caminhos de entrada e saída

- Entrada esperada:
  - `data/raw/train_data.csv`
  - `data/raw/test_data.csv`
- Saída gerada:
  - `data/parquet/train/`
  - `data/parquet/test/`

### 7.3 Execução

```bash
python convertParquet.py
```

Ao final, o script imprime o tempo gasto em cada conversão e confirma a conclusão do processo.

## 8. Reprodutibilidade e Boas Práticas

Para uso acadêmico, recomenda-se:

- registrar versões de bibliotecas e do interpretador Python;
- manter versionamento de dados e scripts de preparação;
- documentar o contrato de consumo da base (formato, partições, nomes de colunas e convenções);
- manter rastreabilidade das versões publicadas para os repositórios consumidores.


## 9. Limitações Atuais

Por definição de escopo, no momento atual este projeto não inclui:

- notebook de análise exploratória;
- treinamento e validação de modelos;

Esses itens devem ser implementados posteriormente.

## 10. Referência

American Express. *American Express - Default Prediction* (Kaggle). Disponível em:
<https://www.kaggle.com/competitions/amex-default-prediction/overview>. Acesso em: 10 maio 2026.

## 11. Licença

Este repositório utiliza a licença definida no arquivo `LICENSE`.