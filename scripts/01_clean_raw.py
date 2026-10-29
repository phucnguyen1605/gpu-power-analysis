"""Clean the raw GPU file into a typed table that keeps every row and column.

Usage:
    python scripts/01_clean_raw.py

Inputs:
    data/raw/All_GPUs.csv
    data/manual/multi_gpu_review.csv     created on the first run, then edited by hand

Outputs (data/interim/):
    gpus_all_clean.parquet               for code: keeps column types
    gpus_all_clean.csv                   for people: open in Excel to inspect
    gpus_all_clean_dictionary.csv        source, type and missing share per column
    multi_gpu_candidates.csv             rule-based candidates before review
"""

import logging

from gpu_power_analysis import clean, paths

logger = logging.getLogger("clean_raw")

# Columns shown before/after parsing in the step-4 log, as a worked example.
EXAMPLE_COLUMNS = {
    "Max_Power": "max_power_w",
    "Memory_Bandwidth": "memory_bandwidth_gb_s",
    "ROPs": "rops",
    "TMUs": "tmus",
    "Memory_Bus": "memory_bus_bit",
    "Release_Date": "release_date",
}


def main() -> None:
    """Run the six cleaning steps and write the outputs."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    paths.INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    raw = clean.read_raw(paths.RAW_GPUS)
    logger.info(
        "Step 1/6  read raw file: %d rows x %d columns", len(raw), raw.shape[1] - 1
    )

    candidates = clean.find_multi_gpu_candidates(raw)
    candidates.to_csv(paths.MULTI_GPU_CANDIDATES, index=False)
    rule_counts = candidates["matched_rules"].str.split(",").explode().value_counts()
    logger.info(
        "Step 2/6  multi-GPU candidates: %d %s", len(candidates), rule_counts.to_dict()
    )

    multi_gpu_ids = clean.apply_review(candidates, paths.MULTI_GPU_REVIEW)
    logger.info("Step 3/6  multi-GPU products after review: %d", len(multi_gpu_ids))

    parsed = clean.parse_columns(raw)
    # Show a single-GPU card with an "a (xN)" ROP count, the least obvious
    # conversion: the suffix multiplies, it does not count GPUs.
    has_multiplier = raw["ROPs"].str.contains(r"\(x", na=False)
    single_gpu = ~raw["gpu_id"].isin(multi_gpu_ids)
    example = raw.index[has_multiplier & single_gpu][0]
    logger.info(
        "Step 4/6  parsed into %d typed columns; example: %s",
        parsed.shape[1],
        raw.at[example, "Name"],
    )
    for raw_column, clean_column in EXAMPLE_COLUMNS.items():
        logger.info(
            "            %-17s %-14r -> %-22s %s",
            raw_column,
            raw.at[example, raw_column],
            clean_column,
            parsed.at[example, clean_column],
        )

    mismatches = clean.check_consistency(parsed)
    logger.info("          consistency between related columns: %s", mismatches)

    flagged = clean.add_flags(parsed, raw, multi_gpu_ids)
    flag_counts = flagged[["is_multi_gpu", "is_unreleased", "is_exact_duplicate"]].sum()
    logger.info("Step 5/6  flags (rows kept, not removed): %s", flag_counts.to_dict())

    dictionary = clean.describe_columns(flagged)
    flagged.to_parquet(paths.CLEAN_ALL_PARQUET, index=False)
    flagged.to_csv(paths.CLEAN_ALL_CSV, index=False)
    dictionary.to_csv(paths.CLEAN_ALL_DICTIONARY, index=False)
    logger.info(
        "Step 6/6  wrote %d rows x %d columns to %s",
        *flagged.shape,
        paths.CLEAN_ALL_PARQUET.relative_to(paths.ROOT),
    )


if __name__ == "__main__":
    main()
