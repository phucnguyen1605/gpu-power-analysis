"""Step-by-step preprocessing of the raw GPU file into the analysis dataset.

Steps:
    1. read_raw          read every column as text, untouched
    2. select_columns    keep only the raw columns the project needs
    3. clean_text        trim whitespace, mark missing values, fix encoding
    4. parse_columns     convert units and types; repair the "(xN)" error
    5. filter_rows       keep released, single, dedicated GPUs
    6. derive_variables  add shared variables; keep the final columns
    7. describe_columns  build the data dictionary

Missing values are kept: every analysis selects its own complete cases, and
the response ``max_power_w`` is never imputed. Outliers, log transforms and
model samples are not handled here; each analysis does them in its own context.
"""

from pathlib import Path

import pandas as pd

from gpu_power_analysis import parsers, paths

# Cell contents that mean "not recorded", in any column.
MISSING_TOKENS = ["", "-"]
UNKNOWN_DATE = "Unknown Release Date"

# Text that was mis-encoded in the raw file (UTF-8 read as Latin-1).
ENCODING_FIXES = {"Â²": "²"}

# Raw columns the project needs. The last six are only used while processing:
# Dedicated to filter, ROPs for the "(xN)" factor, Texture_Rate for the
# efficiency variable, and Boost_Clock, Memory_Bus, Memory_Speed to check the
# repaired values against the texture-rate and bandwidth identities.
USED_COLUMNS = [
    "Name",
    "Manufacturer",
    "Architecture",
    "Release_Date",
    "Notebook_GPU",
    "Max_Power",
    "TMUs",
    "Core_Speed",
    "Memory",
    "Memory_Bandwidth",
    "Memory_Type",
    "Process",
    "Dedicated",
    "ROPs",
    "Texture_Rate",
    "Boost_Clock",
    "Memory_Bus",
    "Memory_Speed",
]

# Name patterns of multi-GPU products (SLI/CrossFire setups, dual-GPU boards,
# AMD Dual Graphics). Every match was checked by hand against the specs in the
# data (a dual board lists twice the TMUs and bus width of its single sibling).
# The rules are narrow on purpose: "Quad", "Dual", "Duo" and "X2" also appear in
# single-GPU names ("Quadro", "Dual-X", "TurboDuo", "IceQ X2" and "Twin X2"
# coolers), "HD 6990M" is a single laptop chip and "ARES X" is a Colorful model.
MULTI_GPU_RULES = {
    "sli_crossfire": r"(?i)\bSLI\b|\bcross ?fire\b|\b[2-4]-way\b",
    "hybrid_graphics": r"\s\+\s",
    "x2_board": r"(?i)\d\s?X2\b|\bGX2\b|PowerColor X2 Edition",
    "quadro_plex": r"\bPlex\b",
    "dual_chip_model": (
        r"(?i)GTX 690|GTX 590|GTX 295|Titan Z\b|HD 5970|HD 6990(?!M)|HD 7990"
        r"|HD 8990|Pro Duo|Devil 13|\bARES\b(?! X)|\bMARS\b|R9 295"
        r"|Tesla K10\b|FirePro S10000|NVS 4[25]0\b"
    ),
}

# A PCIe x16 slot supplies at most 75 W; a card above that needs a connector.
PCIE_SLOT_POWER_W = 75

# Memory types with fewer cards than this are pooled into "Other" so every
# group is large enough for an ANOVA. Cards are counted only where max_power_w
# is known, because those are the only cards an analysis of power can use.
MIN_GROUP_SIZE = 30
OTHER = "Other"

# Role of a column in the analysis.
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


# Final column -> (role, analyses that use it, meaning). Also fixes the order.
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


def read_raw(path: Path) -> pd.DataFrame:
    """Step 1: read every column as text, without changing any value.

    Reading as text means no value is converted before its format is checked
    in step 4. ``gpu_id`` is the row position in the file, kept for tracing.
    """
    raw = pd.read_csv(path, dtype="str", keep_default_na=False)
    return raw.rename_axis("gpu_id").reset_index()


def select_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Step 2: keep the identifier and the raw columns the project needs."""
    return raw[["gpu_id", *USED_COLUMNS]]


def clean_text(selected: pd.DataFrame) -> pd.DataFrame:
    """Step 3: trim whitespace, mark missing values and fix mis-encoded text."""
    text = selected.drop(columns="gpu_id").apply(lambda column: column.str.strip())
    text = text.replace(MISSING_TOKENS, pd.NA)
    text["Release_Date"] = text["Release_Date"].replace(UNKNOWN_DATE, pd.NA)
    for wrong, right in ENCODING_FIXES.items():
        text["Name"] = text["Name"].str.replace(wrong, right, regex=False)
    return pd.concat([selected[["gpu_id"]], text], axis=1)


def parse_columns(selected: pd.DataFrame) -> pd.DataFrame:
    """Step 4: convert units and types; repair TMUs and Memory_Bus.

    Where ROPs read "a (xN)", the source listed TMUs and Memory_Bus divided by
    N: Texture_Rate and Memory_Bandwidth only agree with them once multiplied.
    """
    num = parsers.parse_number
    release_date = parsers.parse_date(selected["Release_Date"])
    multiplier = parsers.parse_multiplier(selected["ROPs"])
    return pd.DataFrame(
        {
            "gpu_id": selected["gpu_id"].astype("int64"),
            "name": selected["Name"],
            "manufacturer": selected["Manufacturer"].astype("category"),
            "architecture": selected["Architecture"],
            "release_year": release_date.dt.year.astype("Int64"),
            "is_dedicated": parsers.parse_yes_no(selected["Dedicated"]),
            "is_notebook": parsers.parse_yes_no(selected["Notebook_GPU"]),
            "max_power_w": num(selected["Max_Power"], "Watts"),
            "tmus": num(selected["TMUs"]) * multiplier,
            "spec_multiplier": multiplier,
            "core_speed_mhz": num(selected["Core_Speed"], "MHz"),
            "boost_clock_mhz": num(selected["Boost_Clock"], "MHz"),
            "memory_mb": num(selected["Memory"], "MB"),
            "memory_bus_bit": num(selected["Memory_Bus"], "Bit") * multiplier,
            "memory_speed_mhz": num(selected["Memory_Speed"], "MHz"),
            "memory_bandwidth_gb_s": parsers.parse_bandwidth(
                selected["Memory_Bandwidth"]
            ),
            "memory_type": selected["Memory_Type"].astype("category"),
            "process_nm": num(selected["Process"], "nm"),
            "texture_rate_gtexel_s": num(selected["Texture_Rate"], "GTexel/s"),
        }
    )


def check_consistency(parsed: pd.DataFrame, tolerance: float = 0.1) -> dict[str, int]:
    """Count rows whose related columns disagree by more than ``tolerance``.

    Two identities hold for real GPUs:
      * texture rate = TMUs x clock (boost clock when listed, else base clock);
      * bandwidth = bus width / 8 x memory clock x k, where the data-rate
        factor k is fixed for each memory type (taken as the type's median).
    A row that breaks one of them carries a data error in one of its columns.
    Texture rates are rounded to whole GTexel/s in the raw file, so a gap of
    up to 0.5 is accepted whatever its relative size.
    """
    clock = parsed["boost_clock_mhz"].fillna(parsed["core_speed_mhz"])
    expected_texture = parsed["tmus"] * clock / 1000
    texture_gap = (parsed["texture_rate_gtexel_s"] - expected_texture).abs()
    data_rate = (
        parsed["memory_bandwidth_gb_s"]
        * 1000
        / (parsed["memory_bus_bit"] / 8 * parsed["memory_speed_mhz"])
    )
    typical = data_rate.groupby(parsed["memory_type"], observed=True).transform(
        "median"
    )
    return {
        "texture_rate_mismatch": int(
            ((texture_gap > 0.5) & (texture_gap > tolerance * expected_texture)).sum()
        ),
        "bandwidth_mismatch": int(((data_rate / typical - 1).abs() > tolerance).sum()),
    }


def match_multi_gpu_rules(names: pd.Series) -> pd.Series:
    """Comma-separated names of the multi-GPU rules each name matches."""
    matches = pd.DataFrame(
        {rule: names.str.contains(pattern) for rule, pattern in MULTI_GPU_RULES.items()}
    ).fillna(False)
    return matches.apply(lambda row: ",".join(row.index[row]), axis=1)


def filter_rows(parsed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 5: keep released, single, dedicated GPUs.

    Returns:
        The kept rows, and a table with the number of rows after each filter.
    """
    names = parsed["name"]
    value_columns = [column for column in parsed.columns if column != "gpu_id"]
    steps = [
        ("all rows", pd.Series(True, index=parsed.index)),
        ("dedicated GPU", parsed["is_dedicated"].fillna(False)),
        ("not a multi-GPU product", match_multi_gpu_rules(names).eq("")),
        ("released", ~names.str.contains(r"(?i)\[not released\]")),
        ("not an exact duplicate", ~parsed.duplicated(subset=value_columns)),
    ]
    keep = pd.Series(True, index=parsed.index)
    counts = []
    for step, condition in steps:
        keep &= condition
        counts.append({"step": step, "rows": int(keep.sum())})
    row_counts = pd.DataFrame(counts)
    row_counts["removed"] = -row_counts["rows"].diff().fillna(0).astype("int64")
    return parsed.loc[keep].reset_index(drop=True), row_counts


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
    """Step 6: add shared variables and keep only the columns some analysis uses."""
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
    """Step 7: data dictionary with role, use, type and missing share per column."""
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
