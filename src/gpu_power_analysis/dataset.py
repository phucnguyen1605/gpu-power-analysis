"""Step-by-step construction of this project's analysis dataset.

Starts from the cleaned, unfiltered table (``clean.py``) and applies the
decisions that belong to this research question: which rows to study, which
variables to keep and which new variables to derive.

Steps:
    1. filter_rows        keep released, single, dedicated GPUs
    2. derive_variables   shared variables; keep only the columns in use
    3. describe_columns   build the data dictionary

Missing values are kept: every analysis selects its own complete cases, and
the response ``max_power_w`` is never imputed. Outliers, log transforms and
model samples are not handled here; each analysis does them in its own context.
"""

from pathlib import Path

import pandas as pd

from gpu_power_analysis import paths

# A PCIe x16 slot supplies at most 75 W; a card above that needs a connector.
PCIE_SLOT_POWER_W = 75

# Memory types with fewer cards than this are pooled into "Other" so every
# group is large enough for an ANOVA. Cards are counted only where max_power_w
# is known, because those are the only cards an analysis of power can use.
MIN_GROUP_SIZE = 30
OTHER = "Other"

# Role of a column in the analysis. Every kept column has a planned use; columns
# no section needs stay in the layer-1 table (gpus_all_clean) only.
IDENTIFIER = "identifier"
RESPONSE = "response"
DERIVED_RESPONSE = "derived_response"
PREDICTOR = "predictor"
GROUPING = "grouping"
TIME = "time"
CLUSTER = "cluster"

# Analyses that use a column, named so the dictionary reads on its own.
DESCRIPTIVE = "descriptive statistics"
PROPORTION = "confidence interval for a proportion"
TWO_GROUPS = "two-group comparison"
ANOVA = "ANOVA"
REGRESSION = "regression"
VALIDATION = "out-of-sample validation"
DISCUSSION = "discussion"
ALL = "all analyses"


def _uses(*analyses: str) -> str:
    return "; ".join(analyses)


# Column -> (role, analyses that use it, meaning). Also fixes the column order.
COLUMNS = {
    "gpu_id": (IDENTIFIER, ALL, "Row position in the raw file (0-based)"),
    "name": (IDENTIFIER, ALL, "Product name"),
    "manufacturer": (
        GROUPING,
        _uses(DESCRIPTIVE, TWO_GROUPS, ANOVA, REGRESSION),
        "GPU vendor",
    ),
    "segment": (
        GROUPING,
        _uses(DESCRIPTIVE, TWO_GROUPS, ANOVA, REGRESSION),
        "Desktop or Laptop",
    ),
    "memory_type": (GROUPING, DESCRIPTIVE, "Memory technology"),
    "memory_type_group": (
        GROUPING,
        _uses(DESCRIPTIVE, ANOVA),
        f"Memory technology; types with under {MIN_GROUP_SIZE} cards of known power "
        f"pooled into '{OTHER}'",
    ),
    "architecture": (
        CLUSTER,
        REGRESSION,
        "GPU chip; cards sharing a chip are not independent observations",
    ),
    "release_year": (TIME, _uses(DESCRIPTIVE, VALIDATION), "Release year"),
    "max_power_w": (RESPONSE, ALL, "Rated maximum power (W)"),
    "needs_aux_power": (
        DERIVED_RESPONSE,
        _uses(DESCRIPTIVE, PROPORTION),
        "max_power_w above the 75 W a PCIe slot supplies",
    ),
    "texture_rate_per_watt": (
        DERIVED_RESPONSE,
        _uses(DESCRIPTIVE, DISCUSSION),
        "Energy efficiency: texture fill rate (GTexel/s) per watt",
    ),
    **{
        column: (PREDICTOR, _uses(DESCRIPTIVE, REGRESSION, VALIDATION), meaning)
        for column, meaning in {
            "tmus": "Texture mapping units",
            "core_speed_mhz": "Base core clock (MHz)",
            "memory_bandwidth_gb_s": "Memory bandwidth (GB/s)",
            "memory_mb": "Video memory (MB)",
            "process_nm": "Manufacturing process (nm)",
        }.items()
    },
}


def filter_rows(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 1: keep released, single, dedicated GPUs.

    Returns:
        The kept rows, and a table with the number of rows after each filter.
    """
    steps = [
        ("all rows", pd.Series(True, index=base.index)),
        ("dedicated GPU", base["is_dedicated"].fillna(False)),
        ("not a multi-GPU product", ~base["is_multi_gpu"]),
        ("released", ~base["is_unreleased"]),
        ("not an exact duplicate", ~base["is_exact_duplicate"]),
    ]
    keep = pd.Series(True, index=base.index)
    counts = []
    for step, condition in steps:
        keep &= condition
        counts.append({"step": step, "rows": int(keep.sum())})
    row_counts = pd.DataFrame(counts)
    row_counts["removed"] = -row_counts["rows"].diff().fillna(0).astype("int64")
    return base.loc[keep].reset_index(drop=True), row_counts


def pool_rare(
    values: pd.Series, min_size: int, other: str, counted: pd.Series | None = None
) -> pd.Series:
    """Replace categories with fewer than ``min_size`` rows by ``other``.

    Args:
        values: Categories to pool.
        min_size: Smallest group size kept as its own category.
        other: Label of the pooled group.
        counted: Rows that count towards a group's size; all rows if omitted.
    """
    counts = (values[counted] if counted is not None else values).value_counts()
    rare = counts.index[counts < min_size]
    pooled = values.astype("str").where(~values.isin(rare), other)
    return pooled.where(values.notna()).astype("category")


def derive_variables(rows: pd.DataFrame) -> pd.DataFrame:
    """Step 2: add shared variables and keep only the columns some section uses."""
    derived = rows.assign(
        segment=rows["is_notebook"]
        .map({True: "Laptop", False: "Desktop"})
        .astype("category"),
        needs_aux_power=rows["max_power_w"] > PCIE_SLOT_POWER_W,
        texture_rate_per_watt=rows["texture_rate_gtexel_s"] / rows["max_power_w"],
        memory_type_group=pool_rare(
            rows["memory_type"],
            MIN_GROUP_SIZE,
            OTHER,
            counted=rows["max_power_w"].notna(),
        ),
    )
    return derived[list(COLUMNS)]


def describe_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    """Step 3: data dictionary with role, use, type and missing share per column."""
    return pd.DataFrame(
        {
            "column": list(COLUMNS),
            "role": [role for role, _, _ in COLUMNS.values()],
            "used_in": [used_in for _, used_in, _ in COLUMNS.values()],
            "dtype": [str(dataset[column].dtype) for column in COLUMNS],
            "missing": [int(dataset[column].isna().sum()) for column in COLUMNS],
            "missing_pct": [
                round(100 * dataset[column].isna().mean(), 1) for column in COLUMNS
            ],
            "description": [description for _, _, description in COLUMNS.values()],
        }
    )


def load_analysis(path: Path = paths.ANALYSIS_PARQUET) -> pd.DataFrame:
    """Read the analysis dataset back with its dtypes (Parquet keeps them)."""
    return pd.read_parquet(path)
