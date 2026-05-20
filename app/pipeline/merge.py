"""
Módulo de Merge de Labels (Polars)
----------------------------------
Objetivo: Unir o dataset tabular gerado na etapa de agregação com os
arquivos de labels originais.
"""

import polars as pl
import logging

logger = logging.getLogger(__name__)

class MescladorLabels:
    def __init__(self, col_cliente: str = 'customer_ID', col_target: str = 'target'):
        self.col_cliente = col_cliente
        self.col_target = col_target

    def mesclar(self, features_path: str, labels_path: str) -> pl.LazyFrame:
        """
        Lê os arquivos de features e labels (mesmo se particionados) de forma Lazy
        e executa um Inner Join.
        """
        logger.info(f"Preparando merge na chave '{self.col_cliente}'...")
        
        # Lê o parquet tabular que geramos na etapa anterior
        df_features = pl.scan_parquet(features_path)
        
        # Lê os parquets de labels (suporta data_*.parquet automaticamente)
        df_labels = pl.scan_parquet(labels_path)
        
        # Faz o Inner Join. O Inner garante que só manteremos clientes
        # que possuem tanto as features calculadas quanto o label definido.
        df_merged = df_features.join(
            df_labels, 
            on=self.col_cliente, 
            how='inner'
        )
        
        return df_merged