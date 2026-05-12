from pathlib import Path
import duckdb
import time

# =========================
# CONFIGURAÇÕES
# =========================

RAW_DIR = Path("data/raw")

TRAIN_LABELS_CSV = RAW_DIR / "train_labels.csv"
TRAIN_CSV = RAW_DIR / "train_data.csv"
TEST_CSV = RAW_DIR / "test_data.csv"

TRAIN_LABELS_OUTPUT = Path("data/parquet/train_labels")
TRAIN_OUTPUT = Path("data/parquet/train")
TEST_OUTPUT = Path("data/parquet/test")

TRAIN_LABELS_OUTPUT.mkdir(parents=True, exist_ok=True)
TRAIN_OUTPUT.mkdir(parents=True, exist_ok=True)
TEST_OUTPUT.mkdir(parents=True, exist_ok=True)

# Quantidade de linhas por row group
# (ajuste conforme RAM)
ROW_GROUP_SIZE = 1_000_000

# =========================
# CONEXÃO DUCKDB
# =========================

con = duckdb.connect(database=":memory:")

# Melhor uso multicore
con.execute("PRAGMA threads=4")

# =========================
# FUNÇÃO DE CONVERSÃO
# =========================

def convert_csv_to_parquet(csv_path, output_dir, dataset_name):

    print(f"\nIniciando conversão: {dataset_name}")
    start = time.time()

    query = f"""
COPY (
    SELECT *
    FROM read_csv_auto(
        '{csv_path}',
        ignore_errors=true,
        sample_size=-1
    )
)
TO '{output_dir}'
(
    FORMAT PARQUET,
    COMPRESSION ZSTD,
    ROW_GROUP_SIZE {ROW_GROUP_SIZE},
    PER_THREAD_OUTPUT,
    OVERWRITE_OR_IGNORE
);
    """

    con.execute(query)

    elapsed = time.time() - start

    print(f"{dataset_name} convertido com sucesso!")
    print(f"Tempo: {elapsed/60:.2f} minutos")


# =========================
# EXECUÇÃO
# =========================

convert_csv_to_parquet(
    TRAIN_LABELS_CSV,
    TRAIN_LABELS_OUTPUT,
    "TRAIN_LABELS"
)

convert_csv_to_parquet(
    TRAIN_CSV,
    TRAIN_OUTPUT,
    "TRAIN"
)

convert_csv_to_parquet(
    TEST_CSV,
    TEST_OUTPUT,
    "TEST"
)

print("\nConversão finalizada.")
