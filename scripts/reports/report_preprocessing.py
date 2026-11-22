"""Compute every number quoted in the preprocessing section of the report.

The numbers are written as LaTeX macros, so the report text is regenerated with
the data instead of being typed by hand.

Usage:
    python scripts/reports/report_preprocessing.py

Outputs (report/generated/):
    preprocessing_numbers.tex           numbers quoted in the text, as macros
    preprocessing_sample_rows.tex       body of the sample-rows table
    preprocessing_missing_by_period.tex body of the missing-by-release-period table
    preprocessing_power_by_group.tex    body of the power-by-group comparison table
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
MISSING_BY_PERIOD_OUTPUT = GENERATED / "preprocessing_missing_by_period.tex"
POWER_BY_GROUP_OUTPUT = GENERATED / "preprocessing_power_by_group.tex"

# Release periods (first year, last year) used to show when power is missing.
# The first and last periods are open-ended: they take the range of the data.
PERIODS = [(None, 2005), (2006, 2008), (2009, 2011), (2012, 2014), (2015, None)]

# Grouping columns whose levels are compared between the analysis dataset and
# every in-scope card with a rated power.
COMPARED_GROUPINGS = ["segment", "manufacturer"]

# Cards shown after preprocessing: an old Nvidia card, an AMD card, a laptop card,
# one with the "(xN)" TMUs repair and one kept without a core clock.
SAMPLE_GPU_IDS = [1721, 1968, 875, 2180, 1399]
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


def is_kept(scoped: pd.DataFrame) -> pd.Series:
    """In-scope rows that step 5 keeps: no required column is missing."""
    return scoped.drop(columns=pre.OPTIONAL_COLUMNS).notna().all(axis=1)


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


def year_range(years: pd.Series) -> str:
    """'2017' for a single year, '2016--2017' (en dash in LaTeX) otherwise."""
    first, last = int(years.min()), int(years.max())
    return str(first) if first == last else f"{first}--{last}"


def decimal(value: float, digits: int = 1) -> str:
    """Number with a fixed number of decimals and a decimal comma."""
    return f"{value:.{digits}f}".replace(".", ",")


def math_decimal(value: float, digits: int = 2) -> str:
    """Decimal comma for use inside LaTeX math, where a bare comma adds a space."""
    return decimal(value, digits).replace(",", "{,}")


def ci_half_width(values: pd.Series) -> float:
    """Half-width of the 95% large-sample (z) confidence interval for the mean."""
    return stats.norm.ppf(0.975) * values.std() / np.sqrt(len(values))


def imputation_effect(analysis: pd.DataFrame) -> dict[str, str]:
    """Compare dropping missing values with filling them by the median.

    Shows what median imputation would do to the mean, the spread and the
    confidence interval of power, to justify not imputing.
    """
    power = analysis["max_power_w"].astype("float64")
    power_filled = power.fillna(power.median())
    dropped = ~is_kept(analysis)
    return {
        "RowsWithPower": int(power.notna().sum()),
        "RowsIncomplete": int(dropped.sum()),
        "RowsIncompletePct": percent(dropped.sum(), len(analysis)),
        "PowerRowsLostByDropAll": int((power.notna() & dropped).sum()),
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


def invalid_power(parsed: pd.DataFrame) -> dict[str, str | int]:
    """Evidence that the Max_Power values set to missing were data-entry errors.

    The other cards on the same chip are the reference: they all list one value.
    """
    invalid = parsed["name"].isin(pre.INVALID_MAX_POWER) & parsed["max_power_w"].isna()
    chips = parsed.loc[invalid, "architecture"]
    siblings = parsed.loc[
        parsed["architecture"].isin(chips) & ~invalid, "max_power_w"
    ].dropna()
    return {
        "InvalidPowerRows": int(invalid.sum()),
        "InvalidPowerSiblings": len(siblings),
        "InvalidPowerSiblingWatts": ", ".join(
            f"{value:.0f}" for value in sorted(siblings.unique())
        ),
    }


def dropped_with_power(scoped: pd.DataFrame) -> dict[str, str | int]:
    """Compare kept rows with rows dropped although their power is known.

    These dropped rows are the only ones a different treatment of missing
    values could bring back, so they show which way the analysis dataset leans.
    """
    complete = is_kept(scoped)
    lost = scoped["max_power_w"].notna() & ~complete
    power = scoped["max_power_w"].astype("float64")
    years = scoped["release_year"].astype("float64")
    power_z, power_p = two_mean_z_test(power[lost], power[complete])
    desktop = scoped["segment"].eq("Desktop")
    known = scoped["max_power_w"].notna()
    return {
        "LostPowerMean": decimal(power[lost].mean()),
        "LostPowerZ": math_decimal(power_z),
        "LostPowerP": p_value(power_p),
        "LostMeanYear": decimal(years[lost].mean()),
        "KeptMeanYear": decimal(years[complete].mean()),
        "FinalPowerMean": decimal(power[complete].mean()),
        "FinalPowerMedian": decimal(power[complete].median(), 0),
        "AllPowerMean": decimal(power[known].mean()),
        "AllPowerMedian": decimal(power[known].median(), 0),
        "FinalDesktopAuxPct": percent(
            scoped.loc[complete & desktop, "needs_aux_power"].sum(),
            (complete & desktop).sum(),
        ),
        "AllDesktopAuxPct": percent(
            scoped.loc[known & desktop, "needs_aux_power"].sum(),
            (known & desktop).sum(),
        ),
    }


def unknown_year(scoped: pd.DataFrame) -> dict[str, str | int]:
    """Row of the missing-by-period table for cards without a release year."""
    rows = scoped[scoped["release_year"].isna()]
    return {
        "UnknownYearRows": len(rows),
        "UnknownYearPowerMissingPct": percent(
            rows["max_power_w"].isna().sum(), len(rows)
        ),
        "UnknownYearDroppedPct": percent((~is_kept(rows)).sum(), len(rows)),
    }


def missing_by_period(scoped: pd.DataFrame) -> str:
    """LaTeX rows: cards, share without power and share dropped, per period."""
    years = scoped["release_year"]
    lines = []
    for first, last in PERIODS:
        first = int(years.min()) if first is None else first
        last = int(years.max()) if last is None else last
        rows = scoped[years.between(first, last).fillna(False)]
        cells = [
            f"{first}--{last}",
            str(len(rows)),
            percent(rows["max_power_w"].isna().sum(), len(rows)),
            percent((~is_kept(rows)).sum(), len(rows)),
        ]
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def power_by_group(scoped: pd.DataFrame) -> str:
    """LaTeX rows: size, mean and median power per group, kept rows vs all rows.

    "All rows" are the in-scope cards with a rated power, complete or not.
    Groups with no kept row (Intel) are left out.
    """
    complete = is_kept(scoped)
    known = scoped["max_power_w"].notna()
    lines = []
    for grouping in COMPARED_GROUPINGS:
        kept_sizes = scoped.loc[complete, grouping].value_counts()
        for level in kept_sizes.index[kept_sizes > 0]:
            in_group = scoped[grouping].eq(level)
            cells = [str(level)]
            for subset in (complete, known):
                power = scoped.loc[subset & in_group, "max_power_w"].astype("float64")
                cells += [
                    str(len(power)),
                    decimal(power.mean()),
                    decimal(power.median(), 0),
                ]
            lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def core_clock(analysis: pd.DataFrame, scoped: pd.DataFrame) -> dict[str, str | int]:
    """Numbers behind leaving the core clock out of the required columns.

    Across chips the clock moves with the process and the release year, so its
    own share cannot be told apart. Within one chip those are fixed, which
    leaves the clock itself: the slope of ln(power) on ln(clock), both taken
    as deviations from the chip's mean, is the within-chip elasticity.
    """
    known = analysis.dropna(subset=["core_speed_mhz"])
    clock = np.log(known["core_speed_mhz"].astype("float64"))
    power = np.log(known["max_power_w"].astype("float64"))
    chip = known["architecture"]
    clock_dev = clock - clock.groupby(chip).transform("mean")
    power_dev = power - power.groupby(chip).transform("mean")
    varied = clock.groupby(chip).transform("nunique") > 1
    slope = (clock_dev * power_dev)[varied].sum() / (clock_dev**2)[varied].sum()
    required = scoped.drop(columns=pre.OPTIONAL_COLUMNS).notna().all(axis=1)
    # Share of the clock's variance explained by process and release year.
    design = np.column_stack(
        [
            np.ones(len(known)),
            known["process_nm"].astype("float64"),
            known["release_year"].astype("float64"),
        ]
    )
    raw_clock = known["core_speed_mhz"].astype("float64").to_numpy()
    fitted = design @ np.linalg.lstsq(design, raw_clock, rcond=None)[0]
    explained = 1 - ((raw_clock - fitted) ** 2).sum() / raw_clock.var() / len(known)
    return {
        "CoreExplainedPct": decimal(100 * explained, 0),
        "CoreMissingInFinal": int(analysis["core_speed_mhz"].isna().sum()),
        "CoreMissingInFinalPct": percent(
            analysis["core_speed_mhz"].isna().sum(), len(analysis)
        ),
        "CoreKnownInFinal": len(known),
        "RowsIfCoreRequired": int((required & scoped["core_speed_mhz"].notna()).sum()),
        "CoreProcessCorr": math_decimal(
            known["core_speed_mhz"]
            .astype("float64")
            .corr(known["process_nm"].astype("float64"))
        ),
        "CoreYearCorr": math_decimal(
            known["core_speed_mhz"]
            .astype("float64")
            .corr(known["release_year"].astype("float64"))
        ),
        "CoreWithinChipSpreadPct": decimal(100 * clock_dev[varied].std(), 0),
        "CoreBetweenChipSpreadPct": decimal(100 * clock.groupby(chip).mean().std(), 0),
        "CoreWithinChipElasticity": decimal(slope),
        "CoreWithinChipCards": int(varied.sum()),
        "CoreWithinChipChips": int(chip[varied].nunique()),
    }


def data_quality(analysis: pd.DataFrame, scoped: pd.DataFrame) -> dict[str, int]:
    """Counts that bound two statements about the data.

    Power values far from the other cards on the same chip are left as they
    are (only an impossible value is removed), so their number is reported.
    """
    power = analysis["max_power_w"].astype("float64")
    by_chip = power.groupby(analysis["architecture"])
    ratio = power / by_chip.transform("median")
    unusual = by_chip.transform("size").ge(3) & ((ratio > 2) | (ratio < 0.5))
    low = scoped["max_power_w"].lt(10).fillna(False)
    clock_known = analysis["core_speed_mhz"].notna()
    kept, with_power = is_kept(scoped), scoped["max_power_w"].notna()

    def vendor_mean(rows: pd.Series, vendor: str) -> float:
        in_group = rows & scoped["manufacturer"].eq(vendor)
        return scoped.loc[in_group, "max_power_w"].astype("float64").mean()

    return {
        # Cards kept only because the clock is optional draw less power.
        "CoreMissingPowerMean": decimal(power[~clock_known].mean()),
        "CoreKnownPowerMean": decimal(power[clock_known].mean()),
        "NvidiaFinalMean": decimal(vendor_mean(kept, "Nvidia")),
        "NvidiaAllMean": decimal(vendor_mean(with_power, "Nvidia")),
        "NvidiaMinusAmdFinal": math_decimal(
            vendor_mean(kept, "Nvidia") - vendor_mean(kept, "AMD"), 1
        ),
        "NvidiaMinusAmdAll": math_decimal(
            vendor_mean(with_power, "Nvidia") - vendor_mean(with_power, "AMD"), 1
        ),
        "MemoryTypeOtherSize": int(analysis["memory_type_group"].eq(pre.OTHER).sum()),
        "PowerFarFromChipMedian": int(unusual.sum()),
        "LowPowerLaptopCards": int((low & scoped["segment"].eq("Laptop")).sum()),
        "RowsBeforeTwoThousandSix": int(analysis["release_year"].lt(2006).sum()),
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
    # The in-scope rows before incomplete ones are dropped, to describe what is lost.
    scoped = pre.derive_variables(pre.filter_rows(parsed, drop_missing=False)[0])
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
        "RepairedYears": year_range(
            parsed.set_index("gpu_id")
            .loc[analysis["gpu_id"]]
            .query("spec_multiplier > 1")["release_year"]
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
        "RowsScoped": counts["not an exact duplicate"],
        "RowsFinal": counts["no missing required value"],
        "IncompleteRemoved": counts["not an exact duplicate"]
        - counts["no missing required value"],
        "IntelFinal": int(analysis["manufacturer"].eq("Intel").sum()),
        "MultiGpuRemoved": counts["dedicated GPU"] - counts["not a multi-GPU product"],
        "UnreleasedRemoved": counts["not a multi-GPU product"] - counts["released"],
        "DuplicatesRemoved": counts["released"] - counts["not an exact duplicate"],
        "LaptopAuxPower": int(
            analysis.loc[analysis["segment"].eq("Laptop"), "needs_aux_power"].sum()
        ),
        "ColsFinal": analysis.shape[1],
        "Desktop": int(analysis["segment"].eq("Desktop").sum()),
        "Laptop": int(analysis["segment"].eq("Laptop").sum()),
        **invalid_power(parsed),
        **missingness(scoped),
        **dropped_with_power(scoped),
        **unknown_year(scoped),
        **core_clock(analysis, scoped),
        **data_quality(analysis, scoped),
        **imputation_effect(scoped),
        **missing_by_column(scoped),
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
    scoped = pd.read_parquet(paths.IN_SCOPE_PARQUET)
    MISSING_BY_PERIOD_OUTPUT.write_text(
        header + missing_by_period(scoped), encoding="utf-8"
    )
    POWER_BY_GROUP_OUTPUT.write_text(header + power_by_group(scoped), encoding="utf-8")
    logger.info(
        "Wrote %d numbers to %s and %d sample rows to %s",
        len(lines),
        NUMBERS_OUTPUT.relative_to(paths.ROOT),
        len(SAMPLE_GPU_IDS),
        SAMPLE_ROWS_OUTPUT.relative_to(paths.ROOT),
    )


if __name__ == "__main__":
    main()
