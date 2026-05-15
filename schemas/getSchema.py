import duckdb

con = duckdb.connect()

schema = con.execute("""
DESCRIBE
SELECT *
FROM 'data/raw/parquet/train/*0.parquet'; # Ajuste o caminho para um dos arquivos Parquet gerados
""").fetchdf()

schema.to_csv("schemas/schema.csv", index=False)