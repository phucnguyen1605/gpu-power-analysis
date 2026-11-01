"""File locations, resolved from the repository root."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW_GPUS = ROOT / "data" / "raw" / "All_GPUs.csv"

INTERIM_DIR = ROOT / "data" / "interim"
CLEAN_ALL_PARQUET = INTERIM_DIR / "gpus_all_clean.parquet"
CLEAN_ALL_CSV = INTERIM_DIR / "gpus_all_clean.csv"
CLEAN_ALL_DICTIONARY = INTERIM_DIR / "gpus_all_clean_dictionary.csv"

PROCESSED_DIR = ROOT / "data" / "processed"
ANALYSIS_PARQUET = PROCESSED_DIR / "gpus_analysis.parquet"
ANALYSIS_CSV = PROCESSED_DIR / "gpus_analysis.csv"
ANALYSIS_DICTIONARY = PROCESSED_DIR / "gpus_analysis_dictionary.csv"
ROW_COUNTS = PROCESSED_DIR / "row_counts.csv"
