"""One-way ANOVA of rated maximum power across memory type groups.

Computes every number, table and figure quoted in the ANOVA part of the
inferential-statistics section. The response is ``max_power_w``; the factor is
``memory_type_group`` without the mixed "Other" group.

Usage:
    python scripts/reports/report_anova.py

Outputs (report/generated/):
    anova_numbers.tex       numbers quoted in the text, as macros
    anova_groups.tex        body of the per-group summary table
    anova_tukey.tex         body of the Tukey HSD pairwise table
    anova_boxplot.pdf       power by memory type group
    anova_qq.pdf            normal Q-Q plot of each group
"""

import itertools
import logging

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway

from gpu_power_analysis import paths
from gpu_power_analysis import preprocess as pre

logger = logging.getLogger("report_anova")

GENERATED = paths.ROOT / "report" / "generated"
NUMBERS_OUTPUT = GENERATED / "anova_numbers.tex"
GROUPS_OUTPUT = GENERATED / "anova_groups.tex"
TUKEY_OUTPUT = GENERATED / "anova_tukey.tex"
BOXPLOT_OUTPUT = GENERATED / "anova_boxplot.pdf"
QQ_OUTPUT = GENERATED / "anova_qq.pdf"

RESPONSE = "max_power_w"
FACTOR = "memory_type_group"
# Groups in order of memory generation; "Other" mixes unrelated types.
GROUPS = ["DDR2", "DDR3", "GDDR3", "GDDR5", "GDDR5X"]
ALPHA = 0.05


def pair_name(first: str, second: str) -> str:
    """Label of a pairwise comparison, read as 'second minus first'."""
    return f"{second}--{first}"


def decimal(value: float, digits: int = 1) -> str:
    """Number with a decimal comma, written {,} so it also prints right in math."""
    return f"{value:.{digits}f}".replace(".", "{,}")


def interval(low: float, high: float, digits: int = 1) -> str:
    """Confidence interval in math mode, so negative bounds get a minus sign."""
    return f"$[{decimal(low, digits)};\\ {decimal(high, digits)}]$"


def p_value(value: float) -> str:
    """P-value with a decimal comma; values near 0 or 1 are written as a bound."""
    if value < 0.001:
        return "$<$ 0,001"
    if value > 0.999:  # would round to 1,000
        return "$>$ 0,999"
    return decimal(value, 3)


def sci(value: float) -> str:
    """Very small p-value in LaTeX math: 3.7e-124 -> 3{,}7 times 10 to the -124."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa.replace('.', '{,}')} \\cdot 10^{{{int(exponent)}}}"


def load_groups() -> pd.DataFrame:
    """Analysis rows of the compared groups, with the factor as ordered text."""
    data = pre.load_analysis()
    columns = [RESPONSE, FACTOR, "segment", "architecture"]
    data = data.loc[data[FACTOR].isin(GROUPS), columns]
    data[RESPONSE] = data[RESPONSE].astype(float)
    data[FACTOR] = pd.Categorical(data[FACTOR].astype(str), categories=GROUPS)
    return data


def classic_anova(data: pd.DataFrame) -> dict[str, float]:
    """Sums of squares, F statistic and p-value of the classical one-way ANOVA."""
    grouped = data.groupby(FACTOR, observed=True)[RESPONSE]
    grand_mean = data[RESPONSE].mean()
    ss_between = float((grouped.size() * (grouped.mean() - grand_mean) ** 2).sum())
    ss_total = float(((data[RESPONSE] - grand_mean) ** 2).sum())
    ss_within = ss_total - ss_between
    k, n = grouped.ngroups, len(data)
    ms_between, ms_within = ss_between / (k - 1), ss_within / (n - k)
    f_stat = ms_between / ms_within
    return {
        "k": k,
        "n": n,
        "ss_between": ss_between,
        "ss_within": ss_within,
        "ss_total": ss_total,
        "ms_between": ms_between,
        "ms_within": ms_within,
        "f": f_stat,
        "f_critical": float(stats.f.ppf(1 - ALPHA, k - 1, n - k)),
        "p": float(stats.f.sf(f_stat, k - 1, n - k)),
        "eta_sq": ss_between / ss_total,
    }


def tukey_hsd(data: pd.DataFrame) -> pd.DataFrame:
    """Tukey HSD pairwise comparison of group means (equal variances assumed)."""
    result = pairwise_tukeyhsd(data[RESPONSE], data[FACTOR].astype(str), alpha=ALPHA)
    pairs = list(itertools.combinations(result.groupsunique, 2))
    return pd.DataFrame(
        {
            "pair": [pair_name(first, second) for first, second in pairs],
            "diff": result.meandiffs,
            "low": result.confint[:, 0],
            "high": result.confint[:, 1],
            "p": result.pvalues,
        }
    )


def games_howell(data: pd.DataFrame) -> pd.DataFrame:
    """Games-Howell pairwise comparison of group means (unequal variances).

    Same columns as ``tukey_hsd``; used as a check when variances differ.
    """
    summary = data.groupby(FACTOR, observed=True)[RESPONSE].agg(["mean", "var", "size"])
    k = len(summary)
    rows = []
    for first, second in itertools.combinations(summary.index, 2):
        a, b = summary.loc[first], summary.loc[second]
        var_a, var_b = a["var"] / a["size"], b["var"] / b["size"]
        diff = b["mean"] - a["mean"]
        se = np.sqrt(var_a + var_b)
        df = (var_a + var_b) ** 2 / (
            var_a**2 / (a["size"] - 1) + var_b**2 / (b["size"] - 1)
        )
        q_critical = stats.studentized_range.ppf(1 - ALPHA, k, df) / np.sqrt(2)
        rows.append(
            {
                "pair": pair_name(first, second),
                "diff": diff,
                "low": diff - q_critical * se,
                "high": diff + q_critical * se,
                "p": stats.studentized_range.sf(abs(diff) / se * np.sqrt(2), k, df),
            }
        )
    return pd.DataFrame(rows)


def group_table(data: pd.DataFrame, shapiro: dict[str, float]) -> str:
    """Rows: group, cards, mean, standard deviation, median, Shapiro-Wilk p-value."""
    grouped = data.groupby(FACTOR, observed=True)[RESPONSE]
    lines = []
    for group, values in grouped:
        lines.append(
            f"{group} & {len(values)} & {decimal(values.mean())} & "
            f"{decimal(values.std())} & {values.median():.0f} & "
            f"{p_value(shapiro[group])} \\\\"
        )
    return "\n".join(lines) + "\n"


def posthoc_table(pairs: pd.DataFrame) -> str:
    """Rows: pair, mean difference, 95% interval, adjusted p-value, decision."""
    lines = []
    for row in pairs.itertuples():
        decision = "Bác bỏ" if row.p < ALPHA else "Không bác bỏ"
        lines.append(
            f"{row.pair} & {decimal(row.diff)} & "
            f"{interval(row.low, row.high)} & "
            f"{p_value(row.p)} & {decision} \\\\"
        )
    return "\n".join(lines) + "\n"


def plot_boxplot(data: pd.DataFrame) -> None:
    """Box plot of power per group, with group means marked."""
    values = [data.loc[data[FACTOR] == g, RESPONSE] for g in GROUPS]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.boxplot(values, tick_labels=GROUPS, showmeans=True, widths=0.55)
    ax.set_xlabel("Nhóm loại bộ nhớ")
    ax.set_ylabel("Công suất tối đa (W)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(BOXPLOT_OUTPUT)
    plt.close(fig)


def plot_qq(data: pd.DataFrame) -> None:
    """Normal Q-Q plot of each group, one panel per group."""
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(10, 3))
    for ax, group in zip(axes, GROUPS, strict=True):
        stats.probplot(data.loc[data[FACTOR] == group, RESPONSE], plot=ax)
        ax.set_title(group, fontsize=12)
        ax.set_xlabel("Phân vị chuẩn", fontsize=10)
        ax.set_ylabel("Phân vị mẫu (W)" if group == GROUPS[0] else "", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.get_lines()[0].set_markersize(2)
    fig.tight_layout()
    fig.savefig(QQ_OUTPUT)
    plt.close(fig)


def missing_effect(data: pd.DataFrame, group: str) -> dict[str, str | int]:
    """Cards of one group kept in the analysis vs every in-scope card with a power.

    Reads the in-scope file written by preprocessing (Mục 3); nothing is changed.
    """
    scoped = pd.read_parquet(paths.IN_SCOPE_PARQUET)
    with_power = scoped.loc[
        scoped[FACTOR].eq(group) & scoped[RESPONSE].notna(), RESPONSE
    ].astype(float)
    kept = data.loc[data[FACTOR] == group, RESPONSE]
    return {
        "GddrThreeKept": len(kept),
        "GddrThreeWithPower": len(with_power),
        "GddrThreeMeanKept": decimal(kept.mean()),
        "GddrThreeMeanAll": decimal(with_power.mean()),
    }


def numbers(
    data: pd.DataFrame, tukey: pd.DataFrame, games: pd.DataFrame
) -> dict[str, str | int]:
    """Every number quoted in the text of the ANOVA section."""
    groups = [data.loc[data[FACTOR] == g, RESPONSE] for g in GROUPS]
    classic = classic_anova(data)
    welch = anova_oneway(data[RESPONSE], data[FACTOR], use_var="unequal")
    levene = stats.levene(*groups, center="median")
    kruskal = stats.kruskal(*groups)
    sizes = data.groupby(FACTOR, observed=True).size()
    sds = data.groupby(FACTOR, observed=True)[RESPONSE].std()
    ddr3 = data.loc[data[FACTOR] == "DDR3", "segment"]
    # Rated power of GDDR5X cards sits on a few fixed values.
    gddr5x_levels = data.loc[data[FACTOR] == "GDDR5X", RESPONSE].value_counts()
    top_levels = sorted(gddr5x_levels.index[:3])
    by_pair = tukey.set_index("pair")
    games_by_pair = games.set_index("pair")
    top_pair, low_pair = pair_name("GDDR5", "GDDR5X"), pair_name("DDR2", "DDR3")
    tukey_significant = tukey["p"] < ALPHA
    games_significant = games["p"] < ALPHA
    return {
        "Rows": classic["n"],
        "Groups": classic["k"],
        "OtherRows": int(pre.load_analysis()[FACTOR].eq("Other").sum()),
        "DfBetween": classic["k"] - 1,
        "DfWithin": classic["n"] - classic["k"],
        "SsBetween": f"{classic['ss_between']:,.0f}".replace(",", "\\,"),
        "SsWithin": f"{classic['ss_within']:,.0f}".replace(",", "\\,"),
        "SsTotal": f"{classic['ss_total']:,.0f}".replace(",", "\\,"),
        "MsBetween": f"{classic['ms_between']:,.0f}".replace(",", "\\,"),
        "MsWithin": f"{classic['ms_within']:,.0f}".replace(",", "\\,"),
        "ClassicF": decimal(classic["f"], 2),
        "ClassicFCritical": decimal(classic["f_critical"], 2),
        "ClassicP": sci(classic["p"]),
        "EtaSq": decimal(classic["eta_sq"], 3),
        "EtaSqPercent": decimal(100 * classic["eta_sq"], 1),
        "WelchF": decimal(welch.statistic, 2),
        "WelchDfOne": f"{welch.df[0]:.0f}",
        "WelchDfTwo": decimal(welch.df[1], 1),
        "WelchFCritical": decimal(stats.f.ppf(1 - ALPHA, *welch.df), 2),
        "WelchP": sci(welch.pvalue),
        "LeveneW": decimal(levene.statistic, 2),
        "LeveneP": sci(levene.pvalue),
        "SdRatio": decimal(sds.max() / sds.min(), 1),
        "SdMaxGroup": sds.idxmax(),
        "SdMinGroup": sds.idxmin(),
        "SizeMin": int(sizes.min()),
        "SizeMinGroup": sizes.idxmin(),
        "SizeMax": int(sizes.max()),
        "SizeOtherMin": int(sizes.drop(sizes.idxmin()).min()),
        "DdrThreeLaptopPercent": f"{100 * ddr3.eq('Laptop').mean():.0f}",
        "GddrFiveXLevels": ", ".join(f"{v:.0f}" for v in top_levels),
        "GddrFiveXLevelCards": int(gddr5x_levels.loc[top_levels].sum()),
        "GddrFiveXCards": int(sizes["GDDR5X"]),
        # Distinct values of the existing column, counted as-is (no cleaning):
        # a rough sign of how many cards share a chip.
        "ArchLabels": int(data["architecture"].nunique()),
        **missing_effect(data, "GDDR3"),
        "GddrFiveXArchLabels": int(
            data.loc[data[FACTOR] == "GDDR5X", "architecture"].nunique()
        ),
        "GddrFiveXMinusGddrFive": f"{by_pair.loc[top_pair, 'diff']:.0f}",
        "GddrFiveXMinusGddrFiveLow": decimal(by_pair.loc[top_pair, "low"], 1),
        "GddrFiveXMinusGddrFiveHigh": decimal(by_pair.loc[top_pair, "high"], 1),
        "DdrThreeMinusDdrTwoLow": decimal(by_pair.loc[low_pair, "low"], 1),
        "DdrThreeMinusDdrTwoHigh": decimal(by_pair.loc[low_pair, "high"], 1),
        "DdrThreeMinusDdrTwoGamesLow": decimal(games_by_pair.loc[low_pair, "low"], 1),
        "DdrThreeMinusDdrTwoGamesHigh": decimal(games_by_pair.loc[low_pair, "high"], 1),
        "KruskalH": decimal(kruskal.statistic, 1),
        "KruskalP": sci(kruskal.pvalue),
        "Pairs": len(tukey),
        "PairsSignificant": int(tukey_significant.sum()),
        "PairsNotSignificant": ", ".join(tukey.loc[~tukey_significant, "pair"]),
        "GamesSignificant": int(games_significant.sum()),
        "GamesNotSignificant": ", ".join(games.loc[~games_significant, "pair"]),
    }


def main() -> None:
    """Run the analysis and write the macros, tables and figures."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = load_groups()
    shapiro = {
        group: stats.shapiro(values).pvalue
        for group, values in data.groupby(FACTOR, observed=True)[RESPONSE]
    }
    tukey = tukey_hsd(data)
    games = games_howell(data)
    values = numbers(data, tukey, games)

    header = "% Generated by scripts/reports/report_anova.py; do not edit by hand.\n"
    lines = [
        f"\\newcommand{{\\Anova{name}}}{{{value}}}" for name, value in values.items()
    ]
    GENERATED.mkdir(parents=True, exist_ok=True)
    NUMBERS_OUTPUT.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    GROUPS_OUTPUT.write_text(header + group_table(data, shapiro), encoding="utf-8")
    TUKEY_OUTPUT.write_text(header + posthoc_table(tukey), encoding="utf-8")
    plot_boxplot(data)
    plot_qq(data)

    for name, value in values.items():
        logger.info("%-20s %s", name, value)
    logger.info("Tukey HSD:\n%s", tukey.round(3).to_string(index=False))
    logger.info("Games-Howell:\n%s", games.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
