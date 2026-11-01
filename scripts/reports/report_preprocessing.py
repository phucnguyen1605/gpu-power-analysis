"""Compute every number quoted in the preprocessing section of the report.

The numbers are written as LaTeX macros, so the report text is regenerated with
the data instead of being typed by hand.

Usage:
    python scripts/report_preprocessing.py

Outputs (report/generated/):
    preprocessing_numbers.tex           numbers quoted in the text, as macros
    preprocessing_sample_rows.tex       body of the sample-rows table
"""

import logging

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
SAMPLE_GPU_IDS = [1721, 1679, 2227, 1434]
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


def texture_mismatches(parsed: pd.DataFrame) -> dict[str, int]:
    """Rows breaking the texture-rate identity before and after the TMUs repair."""
    undone = parsed.assign(tmus=parsed["tmus"] / parsed["spec_multiplier"])
    return {
        "TextureMismatchBefore": pre.check_consistency(undone)["texture_rate_mismatch"],
        "TextureMismatchAfter": pre.check_consistency(parsed)["texture_rate_mismatch"],
    }


def missingness(analysis: pd.DataFrame) -> dict[str, str | int]:
    """Compare cards with and without max_power_w to describe why power is missing."""
    missing = analysis["max_power_w"].isna()
    years = analysis["release_year"].astype("float64")
    year_test = stats.mannwhitneyu(
        years[missing].dropna(), years[~missing].dropna(), alternative="two-sided"
    )
    major = analysis["manufacturer"].isin(["Nvidia", "AMD"])
    maker = analysis.loc[major, "manufacturer"].astype("str")
    maker_test = stats.chi2_contingency(pd.crosstab(maker, missing[major]))
    amd = analysis["manufacturer"].eq("AMD")
    return {
        "PowerMissing": int(missing.sum()),
        "PowerMissingPct": percent(missing.sum(), len(analysis)),
        "MissingMedianYear": int(years[missing].median()),
        "ObservedMedianYear": int(years[~missing].median()),
        "MissingYearP": p_value(year_test.pvalue),
        "MissingAmdPct": percent((missing & amd).sum(), (missing & major).sum()),
        "ObservedAmdPct": percent((~missing & amd).sum(), (~missing & major).sum()),
        "MissingMakerP": p_value(maker_test.pvalue),
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
    parsed = pre.parse_columns(pre.clean_text(selected))
    rows, row_counts = pre.filter_rows(parsed)
    analysis = pre.derive_variables(rows)
    counts = dict(zip(row_counts["step"], row_counts["rows"], strict=True))
    return {
        "RowsRaw": len(raw),
        "ColsRaw": raw.shape[1] - 1,
        "ColsSelected": selected.shape[1] - 1,
        **raw_text_counts(),
        **texture_mismatches(parsed),
        "Integrated": int(parsed["is_dedicated"].eq(False).sum()),
        "UnknownDedicated": int(parsed["is_dedicated"].isna().sum()),
        "RowsDedicated": counts["dedicated GPU"],
        "RowsSingleGpu": counts["not a multi-GPU product"],
        "RowsReleased": counts["released"],
        "RowsFinal": counts["not an exact duplicate"],
        "ColsFinal": analysis.shape[1],
        "Desktop": int(analysis["segment"].eq("Desktop").sum()),
        "Laptop": int(analysis["segment"].eq("Laptop").sum()),
        **missingness(analysis),
        **missing_by_column(analysis),
    }


def main() -> None:
    """Write the macros file and the sample-rows table body."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    header = "% Generated by scripts/report_preprocessing.py; do not edit by hand.\n"
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
