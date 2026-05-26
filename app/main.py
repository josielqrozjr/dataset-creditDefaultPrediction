"""
Pipeline Híbrido de Preparação de Dados (DuckDB + Polars)
---------------------------------------------------------
1. DuckDB: Lê os arquivos brutos particionados e aplica engenharia temporal via SQL.
2. Handoff: Salva um arquivo Parquet intermediário em disco (para evitar OOM na RAM).
   2.1 Polars: Lê o arquivo intermediário (Lazy), aplica agregações matemáticas avançadas
   (trend features) e salva o tabular final.
3. Merge e Split Estratificado: Junta o treino e teste, depois separa novamente mantendo a 
   proporção de inadimplentes.
4. Polars/LGBM: Faz o merge, split estratificado e seleção de features inteligente.
"""

import os
import glob
import duckdb
import polars as pl
import logging
import gc

# Importações das classes do Pipeline
from pipeline.feature_engineering import EngenhariaTemporal
from pipeline.aggregation import AgregadorClientePolars
from pipeline.feature_selection import SelecionadorFeaturesAMEX

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # ---------------------------------------------------------
    # CONFIGURAÇÃO DE CAMINHOS GERAIS
    # ---------------------------------------------------------
    caminho_input_glob = "./../data/raw/parquet/train/data_*.parquet"
    caminho_output_dir = "./../data/processed/"
    caminho_features = "./../data/processed/dataset_final.parquet"
    caminho_labels_glob = "./../data/raw/parquet/train_labels/data_*.parquet"
    caminho_treino_split = "./../data/processed/merge_split/train_80.parquet"
    caminho_saida_selecao = "./../data/processed/selection/train_80_selected.parquet"
    caminho_lista_features = "./../data/processed/selection/selected_features_list.txt"
    
    # Arquivos de saída das Fases 1 e 2
    arquivo_intermediario = os.path.join(caminho_output_dir, "temp_temporal.parquet")
    arquivo_final = os.path.join(caminho_output_dir, "dataset_final.parquet")
    
    os.makedirs(caminho_output_dir, exist_ok=True)
    
    arquivos_encontrados = glob.glob(caminho_input_glob)
    if not arquivos_encontrados:
        logger.error(f"Nenhum arquivo encontrado para o padrão: {caminho_input_glob}")
        return
    logger.info(f"Encontrados {len(arquivos_encontrados)} arquivos particionados.")

    # =========================================================
    # FASE 1: DUCKDB (Engenharia Temporal)
    # =========================================================
    logger.info("=== INICIANDO FASE 1: DUCKDB ===")
    conn = duckdb.connect(':memory:')
    
    try:
        # Extração de Schema
        schema_df = conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{caminho_input_glob}')").df()
        colunas_originais = schema_df['column_name'].tolist()
        
        # Geração da Query SQL
        tabela_leitura = f"read_parquet('{caminho_input_glob}')"
        engenheiro = EngenhariaTemporal()
        sql_temporal = engenheiro.gerar_sql_temporal(tabela_origem=tabela_leitura, colunas_totais=colunas_originais)
        
        # Query de execução e dump para Parquet intermediário
        query_duckdb = f"""
        COPY (
            {sql_temporal}
        ) TO '{arquivo_intermediario}' (FORMAT PARQUET);
        """
        
        logger.info(f"Executando SQL de Engenharia Temporal e salvando dump intermediário...")
        conn.execute(query_duckdb)
        logger.info("Fase 1 (DuckDB) concluída com sucesso.")
        
    except Exception as e:
        logger.exception(f"Erro na Fase 1 (DuckDB): {e}")
        return
    finally:
        conn.close() # Libera a memória do DuckDB
        gc.collect()

    # =========================================================
    # FASE 2: POLARS (Agregação de Clientes com Trend)
    # =========================================================
    logger.info("=== INICIANDO FASE 2: POLARS ===")
    try:
        agregador = AgregadorClientePolars()
        
        # Scan Lazy do arquivo intermediário gerado pelo DuckDB
        logger.info(f"Lendo dump intermediário ({arquivo_intermediario}) de forma Lazy...")
        lazy_df = pl.scan_parquet(arquivo_intermediario)
        
        # Aplicando a transformação
        logger.info("Aplicando agregações matemáticas (estatísticas e trends)...")
        lazy_final = agregador.transformar(lazy_df)
        
        # Executando o motor do Polars e salvando o final
        logger.info(f"Processando e salvando dataset final em: {arquivo_final}")
        lazy_final.sink_parquet(arquivo_final)
        logger.info("Fase 2 (Polars) concluída com sucesso.")
        
    except Exception as e:
        logger.exception(f"Erro na Fase 2 (Polars): {e}")
        return

    # =========================================================
    # LIMPEZA
    # =========================================================
    logger.info("Limpando arquivos intermediários...")
    try:
        if os.path.exists(arquivo_intermediario):
            os.remove(arquivo_intermediario)
            logger.info("Arquivo intermediário removido.")
    except OSError as e:
        logger.warning(f"Não foi possível remover o arquivo intermediário: {e}")

    # =========================================================
    # FASE 3: MERGE E SPLIT ESTRATIFICADO
    # =========================================================
    logger.info("=== INICIANDO FASE 3: MERGE E SPLIT ESTRATIFICADO ===")
    try:
        from pipeline.merge_split import merge_and_split
        merge_and_split(caminho_features, caminho_labels_glob, caminho_output_dir)
        logger.info("Fase 3 (Merge e Split) concluída com sucesso.")
    except Exception as e:
        logger.exception(f"Erro na Fase 3 (Merge e Split Estratificado): {e}")
        return

    # =========================================================
    # FASE 4: FEATURE SELECTION (Somente no Treino)
    # =========================================================
    logger.info("=== INICIANDO FASE 4: FEATURE SELECTION ===")
    try:        
        
        os.makedirs(os.path.dirname(caminho_saida_selecao), exist_ok=True)
        
        logger.info(f"Lendo base de treino (80%) para seleção: {caminho_treino_split}")
        # Usamos read_parquet (em memória) porque a base já está comprimida e agregada
        df_train_80 = pl.read_parquet(caminho_treino_split)
        
        selecionador = SelecionadorFeaturesAMEX()
        df_train_selected = selecionador.selecionar(df_train_80)
        
        logger.info(f"Salvando dataset selecionado em: {caminho_saida_selecao}")
        df_train_selected.write_parquet(caminho_saida_selecao)
        
        # Salva a lista de colunas finais para replicar na base de validação depois
        logger.info("Exportando lista de features selecionadas (.txt)...")
        with open(caminho_lista_features, "w") as f:
            for col in df_train_selected.columns:
                f.write(f"{col}\n")
                
        logger.info("Fase 4 (Feature Selection) concluída com sucesso.")
        
    except Exception as e:
        logger.exception(f"Erro na Fase 4 (Feature Selection): {e}")
        return

    logger.info("=== PIPELINE COMPLETO FINALIZADO COM SUCESSO ===")

if __name__ == "__main__":
    main()