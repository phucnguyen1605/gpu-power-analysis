"""File locations, resolved from the repository root."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW_GPUS = ROOT / "data" / "raw" / "All_GPUs.csv"
MULTI_GPU_REVIEW = ROOT / "data" / "manual" / "multi_gpu_review.csv"

INTERIM_DIR = ROOT / "data" / "interim"
MULTI_GPU_CANDIDATES = INTERIM_DIR / "multi_gpu_candidates.csv"
CLEAN_ALL_PARQUET = INTERIM_DIR / "gpus_all_clean.parquet"
CLEAN_ALL_CSV = INTERIM_DIR / "gpus_all_clean.csv"
CLEAN_ALL_DICTIONARY = INTERIM_DIR / "gpus_all_clean_dictionary.csv"
