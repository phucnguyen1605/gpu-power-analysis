"""Compute every number quoted in the preprocessing section of the report.

The numbers are written as LaTeX macros, so the report text is regenerated with
the data instead of being typed by hand.

Usage:
    python scripts/reports/report_preprocessing.py

Outputs (report/generated/):
    preprocessing_numbers.tex           numbers quoted in the text, as macros
    preprocessing_sample_rows.tex       body of the sample-rows table
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

from gpu_power_analysis import paths
from gpu_power_analysis import preprocess as pre

logger = logging.getLogger("report_preprocessing")

GENERATED = paths.ROOT / "report" / "generated"
NUMBERS_OUTPUT = GENERATED / "preprocessing_numbers.tex"
SAMPLE_ROWS_OUTPUT = GENERATED / "preprocessing_sample_rows.tex"

# Cards shown after preprocessing: an old card, one missing power, one with the
# "(xN)" TMUs repair and one missing core speed.
SAMPLE_GPU_IDS = [1721, 1679, 2180, 1434]
SAMPLE_COLUMNS = {
    "name": "{}",
    "release_year": "{:d}",
    "max_power_w": "{:.0f}",
    "tmus": "{:.0f}",
    "core_speed_mhz": "{:.0f}",
    "memory_bandwidth_gb_s": "{:.1f}",
    "memory_mb": "{:.0f}",
    "process_nm": "{:.0f}",
}


def percent(part: float, whole: float) -> str:
    """Percentage with one decimal and a decimal comma, as written in Vietnamese."""
    return f"{100 * part / whole:.1f}".replace(".", ",")


def p_value(value: float) -> str:
    """P-value with a decimal comma; values below 0.001 are written as a bound."""
    return "$<$ 0,001" if value < 0.001 else "= " + f"{value:.3f}".replace(".", ",")


def raw_text_counts() -> dict[str, int]:
    """Count placeholder strings in the raw file before they are marked missing."""
    text = pd.read_csv(paths.RAW_GPUS, dtype="str", keep_default_na=False)
    text = text.apply(lambda column: column.str.strip())
    return {
        "CoreSpeedDash": int(text["Core_Speed"].eq("-").sum()),
        "CoreSpeedBlank": int(text["Core_Speed"].eq("").sum()),
        "UnknownReleaseDate": int(text["Release_Date"].eq(pre.UNKNOWN_DATE).sum()),
        "BandwidthInMb": int(text["Memory_Bandwidth"].str.endswith("MB/sec").sum()),
        "RopsMultiplier": int(text["ROPs"].str.contains(r"\(x", regex=True).sum()),
    }


def identity_mismatches(parsed: pd.DataFrame) -> dict[str, int]:
    """Rows breaking each identity before and after the "(xN)" repair."""
    undone = parsed.assign(
        tmus=parsed["tmus"] / parsed["spec_multiplier"],
        memory_bus_bit=parsed["memory_bus_bit"] / parsed["spec_multiplier"],
    )
    before = pre.check_consistency(undone)
    after = pre.check_consistency(parsed)
    repaired = parsed["spec_multiplier"].gt(1)
    no_clock = parsed["boost_clock_mhz"].isna() & parsed["core_speed_mhz"].isna()
    bandwidth_before = pre.consistency_flags(undone)["bandwidth_mismatch"]
    bandwidth_after = pre.consistency_flags(parsed)["bandwidth_mismatch"]
    return {
        "TextureUntestable": int((repaired & no_clock).sum()),
        "BandwidthMismatchNotRepaired": int((bandwidth_before & ~repaired).sum()),
        # Flagged only because the faulty rows skewed their memory type's median.
        "BandwidthMismatchShifted": int(
            (bandwidth_before & ~repaired & ~bandwidth_after).sum()
        ),
        "TextureMismatchBefore": before["texture_rate_mismatch"],
        "TextureMismatchAfter": after["texture_rate_mismatch"],
        "BandwidthMismatchBefore": before["bandwidth_mismatch"],
        "BandwidthMismatchAfter": after["bandwidth_mismatch"],
    }


def decimal(value: float, digits: int = 1) -> str:
    """Number with a fixed number of decimals and a decimal comma."""
    return f"{value:.{digits}f}".replace(".", ",")


def math_decimal(value: float, digits: int = 2) -> str:
    """Decimal comma for use inside LaTeX math, where a bare comma adds a space."""
    return decimal(value, digits).replace(",", "{,}")


def ci_half_width(values: pd.Series) -> float:
    """Half-width of the 95% t confidence interval for the mean."""
    n = len(values)
    return stats.t.ppf(0.975, n - 1) * values.std() / np.sqrt(n)


def imputation_effect(analysis: pd.DataFrame) -> dict[str, str]:
    """Compare dropping missing values with filling them by the median.

    Shows what median imputation would do to the mean, the spread and the
    confidence interval of power, to justify not imputing.
    """
    power = analysis["max_power_w"].astype("float64")
    power_filled = power.fillna(power.median())
    predictors = [
        "tmus",
        "core_speed_mhz",
        "memory_bandwidth_gb_s",
        "memory_mb",
        "process_nm",
    ]
    return {
        "RowsWithPower": int(power.notna().sum()),
        "RowsComplete": int(analysis.notna().all(axis=1).sum()),
        "RowsIncomplete": int(analysis.isna().any(axis=1).sum()),
        "RowsIncompletePct": percent(analysis.isna().any(axis=1).sum(), len(analysis)),
        "PowerRowsLostByDropAll": int(
            (power.notna() & analysis.isna().any(axis=1)).sum()
        ),
        "RowsWithPowerAndPredictors": int(
            analysis[["max_power_w", *predictors]].notna().all(axis=1).sum()
        ),
        "PowerMeanDropped": decimal(power.mean()),
        "PowerMeanImputed": decimal(power_filled.mean()),
        "PowerSdDropped": decimal(power.std()),
        "PowerSdImputed": decimal(power_filled.std()),
        "PowerCiDropped": decimal(ci_half_width(power.dropna())),
        "PowerCiImputed": decimal(ci_half_width(power_filled)),
        "PowerAtMedianDropped": percent(
            power.eq(power.median()).sum(), power.notna().sum()
        ),
        "PowerAtMedianImputed": percent(
            power_filled.eq(power.median()).sum(), len(power)
        ),
    }


def two_mean_z_test(first: pd.Series, second: pd.Series) -> tuple[float, float]:
    """Large-sample z test for equal means (two-sided), as taught in the course."""
    se = np.sqrt(first.var() / len(first) + second.var() / len(second))
    z = (first.mean() - second.mean()) / se
    return z, 2 * stats.norm.sf(abs(z))


def two_proportion_z_test(
    hits: tuple[int, int], sizes: tuple[int, int]
) -> tuple[float, float]:
    """Pooled z test for equal proportions (two-sided), as taught in the course."""
    rates = [hit / size for hit, size in zip(hits, sizes, strict=True)]
    pooled = sum(hits) / sum(sizes)
    se = np.sqrt(pooled * (1 - pooled) * (1 / sizes[0] + 1 / sizes[1]))
    z = (rates[0] - rates[1]) / se
    return z, 2 * stats.norm.sf(abs(z))


def missingness(analysis: pd.DataFrame) -> dict[str, str | int]:
    """Compare cards with and without max_power_w to describe why power is missing."""
    missing = analysis["max_power_w"].isna()
    years = analysis["release_year"].astype("float64")
    missing_years = years[missing].dropna()
    observed_years = years[~missing].dropna()
    year_z, year_p = two_mean_z_test(missing_years, observed_years)
    major = analysis["manufacturer"].isin(["Nvidia", "AMD"])
    amd = analysis["manufacturer"].eq("AMD")
    amd_hits = (int((missing & amd).sum()), int((~missing & amd).sum()))
    major_sizes = (int((missing & major).sum()), int((~missing & major).sum()))
    maker_z, maker_p = two_proportion_z_test(amd_hits, major_sizes)
    return {
        "PowerMissing": int(missing.sum()),
        "PowerMissingPct": percent(missing.sum(), len(analysis)),
        "MissingMeanYear": decimal(missing_years.mean()),
        "ObservedMeanYear": decimal(observed_years.mean()),
        "MissingYearZ": math_decimal(year_z),
        "MissingYearP": p_value(year_p),
        "MissingAmdPct": percent(amd_hits[0], major_sizes[0]),
        "ObservedAmdPct": percent(amd_hits[1], major_sizes[1]),
        "MissingMakerZ": math_decimal(maker_z),
        "MissingMakerP": p_value(maker_p),
        "AtiCards": int(analysis["manufacturer"].eq("ATI").sum()),
        "AtiMissing": int((missing & analysis["manufacturer"].eq("ATI")).sum()),
        "IntelCards": int(analysis["manufacturer"].eq("Intel").sum()),
    }


def camel_case(column: str) -> str:
    """Macro-safe name of a column: max_power_w -> MaxPowerW."""
    return "".join(part.capitalize() for part in column.split("_"))


def missing_by_column(analysis: pd.DataFrame) -> dict[str, str]:
    """Missing share of every final column, e.g. MissingPctMaxPowerW."""
    return {
        f"MissingPct{camel_case(column)}": percent(
            analysis[column].isna().sum(), len(analysis)
        )
        for column in analysis.columns
    }


def sample_rows(analysis: pd.DataFrame) -> str:
    """LaTeX rows of the sample cards; missing values are written as NA."""
    rows = analysis.set_index("gpu_id").loc[SAMPLE_GPU_IDS]
    lines = []
    for _, row in rows.iterrows():
        cells = [
            "NA"
            if pd.isna(row[column])
            else template.format(row[column]).replace(".", ",")
            for column, template in SAMPLE_COLUMNS.items()
        ]
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def numbers() -> dict[str, str | int]:
    """Every number the preprocessing section quotes, keyed by macro name."""
    raw = pre.read_raw(paths.RAW_GPUS)
    selected = pre.select_columns(raw)
    cleaned = pre.clean_text(selected)
    parsed = pre.parse_columns(cleaned)
    rows, row_counts = pre.filter_rows(parsed)
    analysis = pre.derive_variables(rows)
    counts = dict(zip(row_counts["step"], row_counts["rows"], strict=True))
    return {
        "RowsRaw": len(raw),
        "ColsRaw": raw.shape[1] - 1,
        "ColsSelected": selected.shape[1] - 1,
        "CellsMarkedMissing": int(cleaned.isna().sum().sum()),
        "ReleaseYearMin": int(analysis["release_year"].min()),
        "ReleaseYearMax": int(analysis["release_year"].max()),
        **raw_text_counts(),
        **identity_mismatches(parsed),
        "EncodingFixed": int(
            selected["Name"]
            .str.contains("|".join(pre.ENCODING_FIXES), regex=True)
            .sum()
        ),
        "RepairedInFinal": int(
            parsed.set_index("gpu_id")
            .loc[analysis["gpu_id"], "spec_multiplier"]
            .gt(1)
            .sum()
        ),
        "NotebookYes": int(parsed["is_notebook"].eq(True).sum()),
        "NotebookNo": int(parsed["is_notebook"].eq(False).sum()),
        "ManufacturerLevels": parsed["manufacturer"].nunique(),
        "ManufacturerNames": ", ".join(parsed["manufacturer"].value_counts().index),
        "MemoryTypeLevels": parsed["memory_type"].nunique(),
        "MemoryTypeMissing": int(parsed["memory_type"].isna().sum()),
        "MemoryTypeLevelsFinal": analysis["memory_type"].nunique(),
        "MemoryTypeGroups": analysis["memory_type_group"].nunique(),
        "MemoryTypeOtherNames": ", ".join(
            sorted(
                analysis.loc[
                    analysis["memory_type_group"].eq(pre.OTHER), "memory_type"
                ].unique()
            )
        ),
        "Integrated": int(parsed["is_dedicated"].eq(False).sum()),
        "UnknownDedicated": int(parsed["is_dedicated"].isna().sum()),
        "RowsDedicated": counts["dedicated GPU"],
        "RowsSingleGpu": counts["not a multi-GPU product"],
        "RowsReleased": counts["released"],
        "RowsFinal": counts["not an exact duplicate"],
        "MultiGpuRemoved": counts["dedicated GPU"] - counts["not a multi-GPU product"],
        "UnreleasedRemoved": counts["not a multi-GPU product"] - counts["released"],
        "DuplicatesRemoved": counts["released"] - counts["not an exact duplicate"],
        "LaptopAuxPower": int(
            analysis.loc[analysis["segment"].eq("Laptop"), "needs_aux_power"].sum()
        ),
        "ColsFinal": analysis.shape[1],
        "Desktop": int(analysis["segment"].eq("Desktop").sum()),
        "Laptop": int(analysis["segment"].eq("Laptop").sum()),
        **missingness(analysis),
        **imputation_effect(analysis),
        **missing_by_column(analysis),
    }


def main() -> None:
    """Write the macros file and the sample-rows table body."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    header = (
        "% Generated by scripts/reports/report_preprocessing.py; do not edit by hand.\n"
    )
    lines = [
        f"\\newcommand{{\\Pre{name}}}{{{value}}}" for name, value in numbers().items()
    ]
    GENERATED.mkdir(parents=True, exist_ok=True)
    NUMBERS_OUTPUT.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    analysis = pre.load_analysis()
    SAMPLE_ROWS_OUTPUT.write_text(header + sample_rows(analysis), encoding="utf-8")
    logger.info(
        "Wrote %d numbers to %s and %d sample rows to %s",
        len(lines),
        NUMBERS_OUTPUT.relative_to(paths.ROOT),
        len(SAMPLE_GPU_IDS),
        SAMPLE_ROWS_OUTPUT.relative_to(paths.ROOT),
    )


if __name__ == "__main__":
    main()
