"""Build this project's analysis dataset from the cleaned, unfiltered table.

Usage:
    python scripts/02_build_dataset.py      (after scripts/01_clean_raw.py)

Input:
    data/interim/gpus_all_clean.parquet

Outputs (data/processed/):
    gpus_analysis.parquet                   for code: keeps column types
    gpus_analysis.csv                       for people: open in Excel to inspect
    gpus_analysis_dictionary.csv            type and missing share per column
    row_counts.csv                          rows left after each filter
"""

import logging

from gpu_power_analysis import clean, dataset, paths

logger = logging.getLogger("build_dataset")


def main() -> None:
    """Run the three dataset steps and write the outputs."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    paths.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    base = clean.load_all_clean()
    rows, row_counts = dataset.filter_rows(base)
    logger.info("Step 1/3  filter rows:")
    for step in row_counts.itertuples():
        logger.info(
            "            %-26s %5d rows  (-%d)", step.step, step.rows, step.removed
        )

    derived = dataset.derive_variables(rows)
    logger.info(
        "Step 2/3  derived variables: desktop %d, laptop %d; needs auxiliary power %d",
        int(derived["segment"].eq("Desktop").sum()),
        int(derived["segment"].eq("Laptop").sum()),
        int(derived["needs_aux_power"].sum()),
    )
    logger.info(
        "            memory type groups: %s",
        derived["memory_type_group"].value_counts().to_dict(),
    )

    analysis = derived
    dictionary = dataset.describe_columns(analysis)
    analysis.to_parquet(paths.ANALYSIS_PARQUET, index=False)
    analysis.to_csv(paths.ANALYSIS_CSV, index=False)
    dictionary.to_csv(paths.ANALYSIS_DICTIONARY, index=False)
    row_counts.to_csv(paths.ROW_COUNTS, index=False)
    logger.info(
        "Step 3/3  wrote %d rows x %d columns to %s",
        *analysis.shape,
        paths.ANALYSIS_PARQUET.relative_to(paths.ROOT),
    )
    logger.info(
        "          max_power_w missing in %d rows (kept, never imputed)",
        int(analysis["max_power_w"].isna().sum()),
    )


if __name__ == "__main__":
    main()
