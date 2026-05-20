"""
Pipeline de Preparação de Dados com DuckDB
------------------------------------------
Este orquestrador lê múltiplos arquivos parquet particionados (data_*.parquet),
extrai o esquema, monta a query dinâmica de engenharia e agregação, 
e executa tudo de forma vetorizada.
"""

import os
import glob
import duckdb
import logging
from pipeline.feature_engineering import EngenhariaTemporal
from pipeline.aggregation import AgregadorCliente

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Usando o padrão com asterisco (*) para ler todas as partições
    caminho_input_glob = "./data/raw/parquet/train/data_*.parquet"
    caminho_output_dir = "./data/processed/"
    caminho_output = os.path.join(caminho_output_dir, "train_tabular_final.parquet")
    
    os.makedirs(caminho_output_dir, exist_ok=True)
    
    # Validação rápida para garantir que o Python encontrou os arquivos
    arquivos_encontrados = glob.glob(caminho_input_glob)
    if not arquivos_encontrados:
        logger.error(f"Nenhum arquivo encontrado para o padrão: {caminho_input_glob}")
        return
    logger.info(f"Encontrados {len(arquivos_encontrados)} arquivos particionados para processamento.")
    
    # Inicia conexão DuckDB em memória
    conn = duckdb.connect(':memory:')
    
    # ---------------------------------------------------------
    # 1. Extração do Esquema (Schema) do Parquet
    # ---------------------------------------------------------
    logger.info(f"Lendo metadados dos arquivos parquet...")
    try:
        # O DuckDB lê o schema do primeiro arquivo do glob pattern automaticamente
        schema_df = conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{caminho_input_glob}')").df()
        colunas_originais = schema_df['column_name'].tolist()
        logger.info(f"Esquema unificado carregado. {len(colunas_originais)} colunas encontradas.")
    except Exception as e:
        logger.error(f"Erro ao ler metadados dos parquets: {e}")
        return

    # ---------------------------------------------------------
    # 2. Geração da Query SQL
    # ---------------------------------------------------------
    # Tabela virtual lida diretamente pelo DuckDB agregando todos os data_*.parquet
    tabela_leitura = f"read_parquet('{caminho_input_glob}')"
    
    engenheiro = EngenhariaTemporal()
    agregador = AgregadorCliente()
    
    # Constrói as partes da query baseadas nas classes anteriores
    sql_temporal = engenheiro.gerar_sql_temporal(tabela_origem=tabela_leitura, colunas_totais=colunas_originais)
    sql_agregacao = agregador.gerar_sql_agregacao(nome_cte_temporal="cte_temporal", colunas_originais=colunas_originais)
    
    # Query Final unificando tudo em CTEs
    query_final = f"""
    WITH cte_temporal AS (
        {sql_temporal}
    )
    -- Processa todas as partições em streaming e salva o resultado final em um único Parquet
    COPY (
        {sql_agregacao}
    ) TO '{caminho_output}' (FORMAT PARQUET);
    """

    # ---------------------------------------------------------
    # 3. Execução no Motor do DuckDB
    # ---------------------------------------------------------
    logger.info("Iniciando execução vetorizada via DuckDB sobre todas as partições. Isso pode levar alguns minutos...")
    try:
        conn.execute(query_final)
        logger.info(f"Sucesso! Dataset processado unificado e salvo em: {caminho_output}")
    except Exception as e:
        logger.error(f"Erro durante a execução do pipeline DuckDB: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()