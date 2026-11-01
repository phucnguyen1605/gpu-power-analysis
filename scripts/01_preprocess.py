"""Preprocess the raw GPU file into the analysis dataset of this project.

Usage:
    python scripts/01_preprocess.py

Input:
    data/raw/All_GPUs.csv

Outputs (data/processed/):
    gpus_analysis.parquet               for code: keeps column types
    gpus_analysis.csv                   for people: open in a spreadsheet
    gpus_analysis_dictionary.csv        role, use, type and missing share per column
    row_counts.csv                      rows left after each filter
"""

import logging

from gpu_power_analysis import paths
from gpu_power_analysis import preprocess as pre

logger = logging.getLogger("preprocess")

# Columns shown before/after parsing in the step-4 log, as a worked example.
EXAMPLE_COLUMNS = {
    "Max_Power": "max_power_w",
    "Memory_Bandwidth": "memory_bandwidth_gb_s",
    "TMUs": "tmus",
    "Memory_Bus": "memory_bus_bit",
    "Release_Date": "release_year",
}


def main() -> None:
    """Run the seven preprocessing steps and write the outputs."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    paths.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw = pre.read_raw(paths.RAW_GPUS)
    logger.info(
        "Step 1/7  read raw file as text: %d rows x %d columns",
        len(raw),
        raw.shape[1] - 1,
    )

    selected = pre.select_columns(raw)
    logger.info(
        "Step 2/7  kept %d of %d columns", selected.shape[1] - 1, raw.shape[1] - 1
    )

    cleaned = pre.clean_text(selected)
    newly_missing = int(cleaned.isna().sum().sum() - selected.isna().sum().sum())
    logger.info(
        "Step 3/7  trimmed text; %d placeholder cells marked as missing", newly_missing
    )

    parsed = pre.parse_columns(cleaned)
    # Show a single-GPU card with an "a (xN)" ROP count, the least obvious
    # conversion: TMUs and Memory_Bus are multiplied back by N.
    single_gpu = pre.match_multi_gpu_rules(cleaned["Name"]).eq("")
    example = parsed.index[parsed["spec_multiplier"].gt(1) & single_gpu][0]
    logger.info(
        "Step 4/7  parsed units and types; example: %s", parsed.at[example, "name"]
    )
    for raw_column, column in EXAMPLE_COLUMNS.items():
        logger.info(
            "            %-17s %-14r -> %-22s %s",
            raw_column,
            cleaned.at[example, raw_column],
            column,
            parsed.at[example, column],
        )
    logger.info(
        "          consistency between related columns: %s",
        pre.check_consistency(parsed),
    )

    rows, row_counts = pre.filter_rows(parsed)
    logger.info("Step 5/7  filter rows:")
    for step in row_counts.itertuples():
        logger.info(
            "            %-26s %5d rows  (-%d)", step.step, step.rows, step.removed
        )

    analysis = pre.derive_variables(rows)
    logger.info(
        "Step 6/7  derived variables: desktop %d, laptop %d; needs auxiliary power %d",
        int(analysis["segment"].eq("Desktop").sum()),
        int(analysis["segment"].eq("Laptop").sum()),
        int(analysis["needs_aux_power"].sum()),
    )
    with_power = analysis.loc[analysis["max_power_w"].notna(), "memory_type_group"]
    logger.info(
        "            memory type groups (cards with power): %s",
        with_power.value_counts().to_dict(),
    )

    dictionary = pre.describe_columns(analysis)
    analysis.to_parquet(paths.ANALYSIS_PARQUET, index=False)
    analysis.to_csv(paths.ANALYSIS_CSV, index=False)
    dictionary.to_csv(paths.ANALYSIS_DICTIONARY, index=False)
    row_counts.to_csv(paths.ROW_COUNTS, index=False)
    logger.info(
        "Step 7/7  wrote %d rows x %d columns to %s; max_power_w missing in %d rows",
        *analysis.shape,
        paths.ANALYSIS_PARQUET.relative_to(paths.ROOT),
        int(analysis["max_power_w"].isna().sum()),
    )


if __name__ == "__main__":
    main()
