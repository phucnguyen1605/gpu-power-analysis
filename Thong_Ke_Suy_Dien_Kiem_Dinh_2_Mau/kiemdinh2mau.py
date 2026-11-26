"""Two-sample comparison of rated maximum power between Desktop and Laptop GPUs.

Computes every number, table and figure quoted in the two-sample test part of
the inferential-statistics section. The response is ``max_power_w``; the
grouping variable is ``segment`` (Desktop vs. Laptop).

Usage:
    python scripts/reports/report_two_sample.py

Outputs (report/generated/):
    two_sample_numbers.tex    numbers quoted in the text, as macros
    two_sample_groups.tex     body of the per-group summary table
    two_sample_tests.tex      body of the two-sample test comparison table
    two_sample_boxplot.pdf    power by segment (Desktop vs. Laptop)
    two_sample_qq.pdf         normal Q-Q plot of each segment
"""

import logging
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    from gpu_power_analysis import paths
    from gpu_power_analysis import preprocess as pre

    ROOT = paths.ROOT
except ImportError:
    paths = None
    pre = None
    ROOT = Path(__file__).resolve().parent

logger = logging.getLogger("report_two_sample")

GENERATED = ROOT / "report" / "generated"
NUMBERS_OUTPUT = GENERATED / "two_sample_numbers.tex"
GROUPS_OUTPUT = GENERATED / "two_sample_groups.tex"
TESTS_OUTPUT = GENERATED / "two_sample_tests.tex"
BOXPLOT_OUTPUT = GENERATED / "two_sample_boxplot.pdf"
QQ_OUTPUT = GENERATED / "two_sample_qq.pdf"

RESPONSE = "max_power_w"
GROUP_COL = "segment"
GROUPS = ["Desktop", "Laptop"]
ALPHA = 0.05
PCIE_LIMIT_W = 75.0


def decimal(value: float, digits: int = 1) -> str:
    """Format a float with a Vietnamese decimal comma safe for math mode."""
    return f"{value:.{digits}f}".replace(".", "{,}")


def integer(value: float | int) -> str:
    """Format an integer with thin-space thousands separators for LaTeX."""
    return f"{int(round(value)):,}".replace(",", "\\,")


def interval(low: float, high: float, digits: int = 1) -> str:
    """Format a confidence interval in math mode."""
    return f"$[{decimal(low, digits)};\\ {decimal(high, digits)}]$"


def p_value(value: float) -> str:
    """Format a p-value with decimal comma; extreme values become bounds."""
    if value < 0.001:
        return "$<$ 0,001"
    if value > 0.999:
        return "$>$ 0,999"
    return decimal(value, 3)


def sci(value: float) -> str:
    """Format a very small p-value in scientific notation."""
    if value == 0.0:
        return "< 10^{-300}"
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa.replace('.', '{,}')} \\cdot 10^{{{int(exponent)}}}"


def load_segments() -> pd.DataFrame:
    """Load cleaned GPU data and retain the columns needed for two-sample tests."""
    if pre is not None:
        data = pre.load_analysis()
    else:
        csv_candidates = [
            ROOT / "data" / "processed" / "gpus_analysis.csv",
            ROOT.parent.parent / "data" / "processed" / "gpus_analysis.csv",
            ROOT / "gpus_analysis.csv",
            Path("gpus_analysis.csv"),
        ]
        csv_path = next((p for p in csv_candidates if p.exists()), None)
        if csv_path is None:
            raise FileNotFoundError("Could not find gpus_analysis.csv")
        data = pd.read_csv(csv_path)

    columns = [RESPONSE, GROUP_COL, "manufacturer", "architecture", "release_year"]
    data = data.loc[data[GROUP_COL].isin(GROUPS), columns].copy()
    data[RESPONSE] = data[RESPONSE].astype(float)
    data[GROUP_COL] = pd.Categorical(data[GROUP_COL].astype(str), categories=GROUPS)
    return data


def welch_test(x1: pd.Series, x2: pd.Series, alpha: float = ALPHA) -> dict[str, float]:
    """Welch's two-sample t-test (unequal variances) with 95% CI and Cohen's d."""
    n1, n2 = len(x1), len(x2)
    m1, m2 = float(x1.mean()), float(x2.mean())
    v1, v2 = float(x1.var(ddof=1)), float(x2.var(ddof=1))
    diff = m1 - m2
    se = float(np.sqrt(v1 / n1 + v2 / n2))
    t_stat = diff / se
    df = float(
        (v1 / n1 + v2 / n2) ** 2
        / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    )
    t_crit = float(stats.t.ppf(1 - alpha / 2, df))
    p_val = float(2 * stats.t.sf(abs(t_stat), df))
    pooled_sd = float(np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)))
    return {
        "diff": diff,
        "se": se,
        "t": t_stat,
        "df": df,
        "t_critical": t_crit,
        "p": p_val,
        "ci_low": diff - t_crit * se,
        "ci_high": diff + t_crit * se,
        "cohen_d": diff / pooled_sd,
    }


def pooled_test(x1: pd.Series, x2: pd.Series, alpha: float = ALPHA) -> dict[str, float]:
    """Classical pooled-variance two-sample t-test for sensitivity comparison."""
    n1, n2 = len(x1), len(x2)
    m1, m2 = float(x1.mean()), float(x2.mean())
    v1, v2 = float(x1.var(ddof=1)), float(x2.var(ddof=1))
    diff = m1 - m2
    df = n1 + n2 - 2
    sp = float(np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / df))
    se = float(sp * np.sqrt(1 / n1 + 1 / n2))
    t_stat = diff / se
    t_crit = float(stats.t.ppf(1 - alpha / 2, df))
    p_val = float(2 * stats.t.sf(abs(t_stat), df))
    return {
        "diff": diff,
        "se": se,
        "t": t_stat,
        "df": float(df),
        "t_critical": t_crit,
        "p": p_val,
        "ci_low": diff - t_crit * se,
        "ci_high": diff + t_crit * se,
    }


def mann_whitney_test(x1: pd.Series, x2: pd.Series) -> dict[str, float]:
    """Two-sided Mann-Whitney U test with probability of superiority."""
    n1, n2 = len(x1), len(x2)
    result = stats.mannwhitneyu(x1, x2, alternative="two-sided")
    u_stat = float(result.statistic)
    prob_sup = u_stat / (n1 * n2)
    rank_biserial = 2 * prob_sup - 1
    return {
        "u": u_stat,
        "p": float(result.pvalue),
        "prob_sup": prob_sup,
        "rank_biserial": rank_biserial,
    }


def log_welch_test(
    x1: pd.Series, x2: pd.Series, alpha: float = ALPHA
) -> dict[str, float]:
    """Welch's t-test on natural-log power, returning geometric mean ratio and CI."""
    res = welch_test(np.log(x1), np.log(x2), alpha=alpha)
    return {
        **res,
        "ratio": float(np.exp(res["diff"])),
        "ratio_low": float(np.exp(res["ci_low"])),
        "ratio_high": float(np.exp(res["ci_high"])),
    }


def architecture_level_check(data: pd.DataFrame) -> dict[str, float | int]:
    """Repeat Welch and Mann-Whitney after averaging power per (segment, architecture)."""
    agg = data.groupby([GROUP_COL, "architecture"], observed=True, as_index=False)[
        RESPONSE
    ].mean()
    ax1 = agg.loc[agg[GROUP_COL] == "Desktop", RESPONSE]
    ax2 = agg.loc[agg[GROUP_COL] == "Laptop", RESPONSE]
    w_res = welch_test(ax1, ax2)
    mw_res = mann_whitney_test(ax1, ax2)
    return {
        "n_desktop": len(ax1),
        "n_laptop": len(ax2),
        "mean_desktop": float(ax1.mean()),
        "mean_laptop": float(ax2.mean()),
        "diff": w_res["diff"],
        "t": w_res["t"],
        "df": w_res["df"],
        "p_welch": w_res["p"],
        "ci_low": w_res["ci_low"],
        "ci_high": w_res["ci_high"],
        "p_mw": mw_res["p"],
    }


def group_table(data: pd.DataFrame, shapiro: dict[str, float]) -> str:
    """Build rows: segment, cards, mean, SD, median, IQR, min--max, Shapiro p."""
    grouped = data.groupby(GROUP_COL, observed=True)[RESPONSE]
    lines = []
    for group, values in grouped:
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        lines.append(
            f"{group} & {integer(len(values))} & {decimal(values.mean())} & "
            f"{decimal(values.std())} & {values.median():.0f} & {iqr:.0f} & "
            f"{values.min():.0f}--{values.max():.0f} & "
            f"{p_value(shapiro[group])} \\\\"
        )
    return "\n".join(lines) + "\n"


def tests_table(
    welch: dict[str, float],
    pooled: dict[str, float],
    mw: dict[str, float],
    log_w: dict[str, float],
    arch: dict[str, float | int],
) -> str:
    """Build rows comparing the primary Welch test with robustness checks."""
    rows = [
        (
            "Welch $t$ (chính, W)",
            f"$t_W = {decimal(welch['t'], 2)}$ ($\\nu = {decimal(welch['df'], 1)}$)",
            f"${decimal(welch['diff'], 1)}$ W",
            interval(welch["ci_low"], welch["ci_high"], 1),
            p_value(welch["p"]),
            "Bác bỏ $H_0$",
        ),
        (
            "$t$ phương sai gộp (W)",
            f"$t = {decimal(pooled['t'], 2)}$ ($\\nu = {integer(pooled['df'])}$)",
            f"${decimal(pooled['diff'], 1)}$ W",
            interval(pooled["ci_low"], pooled["ci_high"], 1),
            p_value(pooled["p"]),
            "Bác bỏ $H_0$",
        ),
        (
            "Mann--Whitney $U$",
            f"$U = {integer(mw['u'])}$",
            f"$\\hat{{P}}(X_1 > X_2) = {decimal(mw['prob_sup'], 3)}$",
            "--",
            p_value(mw["p"]),
            "Bác bỏ $H_0$",
        ),
        (
            "Welch $t$ trên $\\ln P$",
            f"$t_W = {decimal(log_w['t'], 2)}$ ($\\nu = {decimal(log_w['df'], 1)}$)",
            f"Tỉ số ${decimal(log_w['ratio'], 2)}\\times$",
            interval(log_w["ratio_low"], log_w["ratio_high"], 2),
            p_value(log_w["p"]),
            "Bác bỏ $H_0$",
        ),
        (
            "Welch $t$ mức kiến trúc (W)",
            f"$t_W = {decimal(arch['t'], 2)}$ ($\\nu = {decimal(arch['df'], 1)}$)",
            f"${decimal(arch['diff'], 1)}$ W",
            interval(arch["ci_low"], arch["ci_high"], 1),
            p_value(arch["p_welch"]),
            "Bác bỏ $H_0$",
        ),
    ]
    return "\n".join(" & ".join(r) + " \\\\" for r in rows) + "\n"


def plot_boxplot(data: pd.DataFrame) -> None:
    """Side-by-side box plot of Desktop vs. Laptop power with means marked."""
    values = [data.loc[data[GROUP_COL] == g, RESPONSE] for g in GROUPS]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.boxplot(values, showmeans=True, widths=0.45)
    ax.set_xticks(range(1, len(GROUPS) + 1))
    ax.set_xticklabels(GROUPS)
    ax.axhline(
        PCIE_LIMIT_W,
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        label="Ngưỡng 75 W (PCIe)",
    )
    ax.set_xlabel("Phân khúc thiết bị")
    ax.set_ylabel("Công suất tối đa (W)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(BOXPLOT_OUTPUT)
    plt.close(fig)


def plot_qq(data: pd.DataFrame) -> None:
    """Normal Q-Q plot of Desktop and Laptop power, one panel per segment."""
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(7.6, 3.3))
    for ax, group in zip(axes, GROUPS, strict=True):
        stats.probplot(data.loc[data[GROUP_COL] == group, RESPONSE], plot=ax)
        ax.set_title(group, fontsize=11)
        ax.set_xlabel("Phân vị chuẩn", fontsize=10)
        ax.set_ylabel("Phân vị mẫu (W)" if group == GROUPS[0] else "", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.get_lines()[0].set_markersize(2.5)
    fig.tight_layout()
    fig.savefig(QQ_OUTPUT)
    plt.close(fig)


def numbers(
    data: pd.DataFrame,
    welch: dict[str, float],
    pooled: dict[str, float],
    mw: dict[str, float],
    log_w: dict[str, float],
    arch: dict[str, float | int],
) -> dict[str, str | int]:
    """Collect every number quoted in the prose of the two-sample section."""
    desktop = data.loc[data[GROUP_COL] == "Desktop", RESPONSE]
    laptop = data.loc[data[GROUP_COL] == "Laptop", RESPONSE]
    levene = stats.levene(desktop, laptop, center="median")
    shapiro_d = stats.shapiro(desktop)
    shapiro_l = stats.shapiro(laptop)

    desk_arch = set(data.loc[data[GROUP_COL] == "Desktop", "architecture"])
    lap_arch = set(data.loc[data[GROUP_COL] == "Laptop", "architecture"])

    nv = data[data["manufacturer"] == "Nvidia"]
    nv_w = welch_test(
        nv.loc[nv[GROUP_COL] == "Desktop", RESPONSE],
        nv.loc[nv[GROUP_COL] == "Laptop", RESPONSE],
    )
    amd = data[data["manufacturer"] == "AMD"]
    amd_w = welch_test(
        amd.loc[amd[GROUP_COL] == "Desktop", RESPONSE],
        amd.loc[amd[GROUP_COL] == "Laptop", RESPONSE],
    )

    lap_q1, lap_q3 = laptop.quantile(0.25), laptop.quantile(0.75)
    lap_upper = lap_q3 + 1.5 * (lap_q3 - lap_q1)

    return {
        "Rows": integer(len(data)),
        "DesktopN": integer(len(desktop)),
        "LaptopN": integer(len(laptop)),
        "DesktopMean": decimal(desktop.mean(), 1),
        "LaptopMean": decimal(laptop.mean(), 1),
        "DesktopSd": decimal(desktop.std(), 1),
        "LaptopSd": decimal(laptop.std(), 1),
        "DesktopMedian": f"{desktop.median():.0f}",
        "LaptopMedian": f"{laptop.median():.0f}",
        "DesktopOverSeventyFivePct": decimal(100 * (desktop > PCIE_LIMIT_W).mean(), 1),
        "LaptopOverSeventyFivePct": decimal(100 * (laptop > PCIE_LIMIT_W).mean(), 1),
        "LaptopUpperFence": f"{lap_upper:.0f}",
        "LaptopUpperOutliers": int((laptop > lap_upper).sum()),
        "SdRatio": decimal(desktop.std() / laptop.std(), 1),
        "SizeRatio": decimal(len(desktop) / len(laptop), 1),
        "ShapiroDesktopW": decimal(shapiro_d.statistic, 3),
        "ShapiroDesktopP": sci(shapiro_d.pvalue),
        "ShapiroLaptopW": decimal(shapiro_l.statistic, 3),
        "ShapiroLaptopP": sci(shapiro_l.pvalue),
        "LeveneF": decimal(levene.statistic, 2),
        "LeveneDfTwo": integer(len(data) - 2),
        "LeveneP": sci(levene.pvalue),
        "WelchDiff": decimal(welch["diff"], 1),
        "WelchSe": decimal(welch["se"], 2),
        "WelchT": decimal(welch["t"], 2),
        "WelchDf": decimal(welch["df"], 1),
        "WelchTCritical": decimal(welch["t_critical"], 2),
        "WelchP": sci(welch["p"]),
        "WelchCiLow": decimal(welch["ci_low"], 1),
        "WelchCiHigh": decimal(welch["ci_high"], 1),
        "CohenD": decimal(welch["cohen_d"], 2),
        "PooledSe": decimal(pooled["se"], 2),
        "PooledT": decimal(pooled["t"], 2),
        "PooledDf": integer(pooled["df"]),
        "PooledP": sci(pooled["p"]),
        "PooledCiLow": decimal(pooled["ci_low"], 1),
        "PooledCiHigh": decimal(pooled["ci_high"], 1),
        "MwU": integer(mw["u"]),
        "MwP": sci(mw["p"]),
        "MwProbSupPct": decimal(100 * mw["prob_sup"], 1),
        "MwRankBiserial": decimal(mw["rank_biserial"], 3),
        "LogWelchT": decimal(log_w["t"], 2),
        "LogWelchDf": decimal(log_w["df"], 1),
        "LogWelchP": sci(log_w["p"]),
        "LogRatio": decimal(log_w["ratio"], 2),
        "LogRatioLow": decimal(log_w["ratio_low"], 2),
        "LogRatioHigh": decimal(log_w["ratio_high"], 2),
        "ArchTotal": int(data["architecture"].nunique()),
        "ArchDesktop": int(arch["n_desktop"]),
        "ArchLaptop": int(arch["n_laptop"]),
        "ArchShared": len(desk_arch & lap_arch),
        "ArchMeanDesktop": decimal(arch["mean_desktop"], 1),
        "ArchMeanLaptop": decimal(arch["mean_laptop"], 1),
        "ArchDiff": decimal(arch["diff"], 1),
        "ArchWelchT": decimal(arch["t"], 2),
        "ArchWelchDf": decimal(arch["df"], 1),
        "ArchWelchP": sci(arch["p_welch"]),
        "ArchCiLow": decimal(arch["ci_low"], 1),
        "ArchCiHigh": decimal(arch["ci_high"], 1),
        "NvidiaDiff": decimal(nv_w["diff"], 1),
        "NvidiaWelchT": decimal(nv_w["t"], 2),
        "AmdDiff": decimal(amd_w["diff"], 1),
        "AmdWelchT": decimal(amd_w["t"], 2),
    }


def main() -> None:
    """Run the two-sample analysis and write macros, tables and figures."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = load_segments()
    desktop = data.loc[data[GROUP_COL] == "Desktop", RESPONSE]
    laptop = data.loc[data[GROUP_COL] == "Laptop", RESPONSE]

    shapiro = {
        group: float(stats.shapiro(values).pvalue)
        for group, values in data.groupby(GROUP_COL, observed=True)[RESPONSE]
    }
    welch = welch_test(desktop, laptop)
    pooled = pooled_test(desktop, laptop)
    mw = mann_whitney_test(desktop, laptop)
    log_w = log_welch_test(desktop, laptop)
    arch = architecture_level_check(data)
    values = numbers(data, welch, pooled, mw, log_w, arch)

    header = "% Generated by scripts/reports/report_two_sample.py; do not edit by hand.\n"
    macro_lines = [
        f"\\newcommand{{\\TwoSample{name}}}{{{value}}}"
        for name, value in values.items()
    ]
    GENERATED.mkdir(parents=True, exist_ok=True)
    NUMBERS_OUTPUT.write_text(header + "\n".join(macro_lines) + "\n", encoding="utf-8")
    GROUPS_OUTPUT.write_text(header + group_table(data, shapiro), encoding="utf-8")
    TESTS_OUTPUT.write_text(
        header + tests_table(welch, pooled, mw, log_w, arch), encoding="utf-8"
    )
    plot_boxplot(data)
    plot_qq(data)

    for name, value in values.items():
        logger.info("%-26s %s", name, value)


if __name__ == "__main__":
    main()