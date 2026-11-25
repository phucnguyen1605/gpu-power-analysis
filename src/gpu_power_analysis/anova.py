"""Two-way factorial ANOVA for GPU power analysis.

Provides hypothesis definitions, cleaned-sample validation, Type II/III
unbalanced two-way ANOVA with assumption checks, Fisher's LSD and Tukey HSD
post-hoc tests, simple-effects Welch t-tests, and dynamic conclusions.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations, product
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

MIN_CELL_COUNT = 2
MIN_SHAPIRO_N = 3
SD_RATIO_WARN_THRESHOLD = 2.0


@dataclass(frozen=True)
class TwoWayHypothesis:
    """Formal research question and hypothesis pairs for a two-way ANOVA."""

    research_question: str
    null_a: str
    alt_a: str
    null_b: str
    alt_b: str
    null_ab: str
    alt_ab: str
    alpha: float = 0.05


@dataclass(frozen=True)
class TwoWayAnovaResult:
    """Complete output of a two-way factorial ANOVA run."""

    response: str
    factor_a: str
    factor_b: str
    alpha: float
    ss_type: Literal["II", "III"]
    n_obs: int
    cell_summary: pd.DataFrame
    anova_table: pd.DataFrame
    levene_stat: float
    levene_p: float
    sd_ratio: float
    shapiro_residuals_stat: float
    shapiro_residuals_p: float
    cell_normality: pd.DataFrame
    tukey_table: pd.DataFrame
    lsd_table: pd.DataFrame
    simple_effects: pd.DataFrame
    residuals: pd.Series


def build_gpu_power_hypotheses(
    response_label: str = "rated maximum power (W)",
    alpha: float = 0.05,
) -> TwoWayHypothesis:
    """Build the research question and hypotheses for manufacturer x segment ANOVA."""
    if not 0 < alpha < 1:
        msg = f"alpha must lie strictly in (0, 1), got {alpha}."
        raise ValueError(msg)
    return TwoWayHypothesis(
        research_question=(
            f"Do manufacturer (Nvidia vs AMD) and market segment (Desktop vs Laptop) "
            f"affect GPU {response_label}, and does the manufacturer difference depend "
            f"on segment (interaction effect)?"
        ),
        null_a=(
            f"H0a: tau_1 = ... = tau_a = 0 (no main effect of manufacturer: "
            f"Nvidia vs AMD on {response_label})."
        ),
        alt_a=(
            f"H1a: At least one tau_i != 0 (manufacturer affects {response_label})."
        ),
        null_b=(
            f"H0b: beta_1 = ... = beta_b = 0 (no main effect of segment: "
            f"Desktop vs Laptop on {response_label})."
        ),
        alt_b=(
            f"H1b: At least one beta_j != 0 (segment affects {response_label})."
        ),
        null_ab=(
            f"H0ab: (tau*beta)_ij = 0 for all i, j (no interaction between "
            f"manufacturer and segment on {response_label})."
        ),
        alt_ab=(
            f"H1ab: At least one (tau*beta)_ij != 0 (interaction exists between "
            f"manufacturer and segment on {response_label})."
        ),
        alpha=alpha,
    )


def _sorted_levels(series: pd.Series) -> list[Any]:
    """Return sorted distinct levels from a non-empty Series."""
    return sorted(series.unique().tolist(), key=lambda val: (str(type(val)), val))


def _check_factorial_cells(
    data: pd.DataFrame,
    factor_a: str,
    factor_b: str,
) -> tuple[list[Any], list[Any]]:
    """Verify at least two levels per factor and >= 2 observations in every cell."""
    levels_a = _sorted_levels(data[factor_a])
    levels_b = _sorted_levels(data[factor_b])
    if len(levels_a) < 2 or len(levels_b) < 2:
        msg = (
            f"Both factors need at least 2 levels; got {factor_a}={levels_a!r}, "
            f"{factor_b}={levels_b!r}."
        )
        raise ValueError(msg)

    counts = data.groupby([factor_a, factor_b], observed=True).size()
    bad_cells = [
        (a_val, b_val, int(counts.get((a_val, b_val), 0)))
        for a_val, b_val in product(levels_a, levels_b)
        if int(counts.get((a_val, b_val), 0)) < MIN_CELL_COUNT
    ]
    if bad_cells:
        detail = ", ".join(
            f"({a_val}, {b_val}): n={cnt}" for a_val, b_val, cnt in bad_cells
        )
        msg = (
            f"Every ({factor_a}, {factor_b}) cell requires at least "
            f"{MIN_CELL_COUNT} observations; found {detail}."
        )
        raise ValueError(msg)
    return levels_a, levels_b


def prepare_two_way_sample(
    data: pd.DataFrame,
    response: str,
    factor_a: str,
    factor_b: str,
    *,
    include_a: Sequence[str] | None = None,
    include_b: Sequence[str] | None = None,
    log_transform: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Filter, validate, and optionally log-transform a cleaned DataFrame for ANOVA."""
    required = [response, factor_a, factor_b]
    missing_cols = [col for col in required if col not in data.columns]
    if missing_cols:
        msg = f"Missing required columns: {missing_cols}."
        raise ValueError(msg)

    sample = data[required].copy()
    if include_a is not None:
        sample = sample[sample[factor_a].isin(include_a)]
    if include_b is not None:
        sample = sample[sample[factor_b].isin(include_b)]
    sample = sample.dropna(subset=required).reset_index(drop=True)

    for col in (factor_a, factor_b):
        if isinstance(sample[col].dtype, pd.CategoricalDtype):
            sample[col] = sample[col].cat.remove_unused_categories()

    _check_factorial_cells(sample, factor_a, factor_b)

    numeric_y = sample[response].astype("float64")
    if log_transform:
        if (numeric_y <= 0).any():
            msg = (
                f"Cannot apply log transform: {response} contains non-positive values."
            )
            raise ValueError(msg)
        target_col = f"log_{response}"
        sample[target_col] = np.log(numeric_y)
        return sample, target_col

    sample[response] = numeric_y
    return sample, response


def _effects_matrix(series: pd.Series, levels: Sequence[Any]) -> np.ndarray:
    """Build sum-to-zero effects coding columns for a categorical factor."""
    n = len(series)
    k = len(levels)
    index_map = {level: idx for idx, level in enumerate(levels)}
    mat = np.zeros((n, k - 1), dtype="float64")
    for row_idx, val in enumerate(series):
        pos = index_map[val]
        if pos < k - 1:
            mat[row_idx, pos] = 1.0
        else:
            mat[row_idx, :] = -1.0
    return mat


def _ols_sse(design: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Return residual sum of squares and residual vector for OLS design."""
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ beta
    return float(np.sum(residuals**2)), residuals


def _compute_anova_table(
    y: np.ndarray,
    data: pd.DataFrame,
    factor_a: str,
    factor_b: str,
    levels_a: Sequence[Any],
    levels_b: Sequence[Any],
    alpha: float,
    ss_type: Literal["II", "III"],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Compute Type II or Type III two-way ANOVA table and full-model residuals."""
    n = len(y)
    x_int = np.ones((n, 1), dtype="float64")
    x_a = _effects_matrix(data[factor_a], levels_a)
    x_b = _effects_matrix(data[factor_b], levels_b)
    x_ab = np.column_stack(
        [
            x_a[:, i] * x_b[:, j]
            for i in range(x_a.shape[1])
            for j in range(x_b.shape[1])
        ]
    )

    x_full = np.column_stack([x_int, x_a, x_b, x_ab])
    df_err = n - x_full.shape[1]
    if df_err <= 0:
        msg = f"Residual degrees of freedom must be positive, got {df_err}."
        raise ValueError(msg)

    sse_full, residuals = _ols_sse(x_full, y)
    ms_err = sse_full / df_err

    sse_no_ab, _ = _ols_sse(np.column_stack([x_int, x_a, x_b]), y)
    ss_ab = sse_no_ab - sse_full

    if ss_type == "III":
        sse_no_a, _ = _ols_sse(np.column_stack([x_int, x_b, x_ab]), y)
        sse_no_b, _ = _ols_sse(np.column_stack([x_int, x_a, x_ab]), y)
        ss_a = sse_no_a - sse_full
        ss_b = sse_no_b - sse_full
    else:
        sse_only_b, _ = _ols_sse(np.column_stack([x_int, x_b]), y)
        sse_only_a, _ = _ols_sse(np.column_stack([x_int, x_a]), y)
        ss_a = sse_only_b - sse_no_ab
        ss_b = sse_only_a - sse_no_ab

    df_a = len(levels_a) - 1
    df_b = len(levels_b) - 1
    df_ab = df_a * df_b
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    rows: list[dict[str, Any]] = []
    for source, ss_val, df_val in [
        (factor_a, ss_a, df_a),
        (factor_b, ss_b, df_b),
        (f"{factor_a}:{factor_b}", ss_ab, df_ab),
    ]:
        ms_val = ss_val / df_val
        f_stat = ms_val / ms_err
        f_crit = float(stats.f.isf(alpha, df_val, df_err))
        p_val = float(stats.f.sf(f_stat, df_val, df_err))
        denom = ss_val + sse_full
        partial_eta_sq = ss_val / denom if denom > 0 else 0.0
        rows.append(
            {
                "source": source,
                "ss": ss_val,
                "df": df_val,
                "ms": ms_val,
                "f_stat": f_stat,
                "f_crit": f_crit,
                "p_value": p_val,
                "partial_eta_sq": partial_eta_sq,
                "reject_h0": p_val < alpha,
            }
        )

    rows.append(
        {
            "source": "Residual",
            "ss": sse_full,
            "df": df_err,
            "ms": ms_err,
            "f_stat": np.nan,
            "f_crit": np.nan,
            "p_value": np.nan,
            "partial_eta_sq": np.nan,
            "reject_h0": pd.NA,
        }
    )
    rows.append(
        {
            "source": "Total",
            "ss": ss_tot,
            "df": n - 1,
            "ms": np.nan,
            "f_stat": np.nan,
            "f_crit": np.nan,
            "p_value": np.nan,
            "partial_eta_sq": np.nan,
            "reject_h0": pd.NA,
        }
    )
    return pd.DataFrame(rows), residuals


def _compute_cell_tables(
    data: pd.DataFrame,
    response: str,
    factor_a: str,
    factor_b: str,
    levels_a: Sequence[Any],
    levels_b: Sequence[Any],
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[np.ndarray]]:
    """Build per-cell descriptive stats, Shapiro-Wilk checks, and raw arrays."""
    summary_rows: list[dict[str, Any]] = []
    normality_rows: list[dict[str, Any]] = []
    cell_labels: list[str] = []
    cell_arrays: list[np.ndarray] = []

    for a_val, b_val in product(levels_a, levels_b):
        mask = data[factor_a].eq(a_val) & data[factor_b].eq(b_val)
        arr = data.loc[mask, response].to_numpy(dtype="float64")
        cell_labels.append(f"{a_val}-{b_val}")
        cell_arrays.append(arr)

        summary_rows.append(
            {
                factor_a: a_val,
                factor_b: b_val,
                "count": len(arr),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)),
                "median": float(np.median(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        )
        if len(arr) >= MIN_SHAPIRO_N:
            w_stat, p_val = stats.shapiro(arr)
            w_val, p_float = float(w_stat), float(p_val)
            normal_flag: bool | Any = p_float >= alpha
        else:
            w_val, p_float = np.nan, np.nan
            normal_flag = pd.NA
        normality_rows.append(
            {
                factor_a: a_val,
                factor_b: b_val,
                "n": len(arr),
                "shapiro_w": w_val,
                "p_value": p_float,
                "normal_at_alpha": normal_flag,
            }
        )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(normality_rows),
        cell_labels,
        cell_arrays,
    )


def _compute_tukey_table(
    cell_labels: Sequence[str],
    cell_arrays: Sequence[np.ndarray],
    alpha: float,
) -> pd.DataFrame:
    """Run Tukey HSD across all cells and return a tidy comparison DataFrame."""
    tukey = stats.tukey_hsd(*cell_arrays)
    ci = tukey.confidence_interval(confidence_level=1.0 - alpha)
    rows: list[dict[str, Any]] = []
    for i, j in combinations(range(len(cell_labels)), 2):
        p_val = float(tukey.pvalue[i, j])
        rows.append(
            {
                "group_1": cell_labels[i],
                "group_2": cell_labels[j],
                "diff": float(tukey.statistic[i, j]),
                "ci_low": float(ci.low[i, j]),
                "ci_high": float(ci.high[i, j]),
                "p_value": p_val,
                "reject_h0": p_val < alpha,
            }
        )
    return pd.DataFrame(rows)


def _compute_lsd_table(
    cell_labels: Sequence[str],
    cell_arrays: Sequence[np.ndarray],
    mse: float,
    df_err: int,
    alpha: float,
) -> pd.DataFrame:
    """Run Fisher's LSD pairwise comparisons for unequal cell sizes."""
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df_err))
    rows: list[dict[str, Any]] = []
    for i, j in combinations(range(len(cell_labels)), 2):
        arr_i = cell_arrays[i]
        arr_j = cell_arrays[j]
        diff = float(np.mean(arr_i) - np.mean(arr_j))
        abs_diff = abs(diff)
        se = float(np.sqrt(mse * (1.0 / len(arr_i) + 1.0 / len(arr_j))))
        lsd_crit = t_crit * se
        t_stat = abs_diff / se if se > 0 else (0.0 if abs_diff == 0.0 else np.inf)
        p_val = float(2.0 * stats.t.sf(t_stat, df_err)) if np.isfinite(t_stat) else 0.0
        rows.append(
            {
                "group_1": cell_labels[i],
                "group_2": cell_labels[j],
                "diff": diff,
                "abs_diff": abs_diff,
                "lsd_crit": lsd_crit,
                "t_crit": t_crit,
                "p_value": p_val,
                "reject_h0": bool(abs_diff > lsd_crit),
            }
        )
    return pd.DataFrame(rows)


def _compute_simple_effects(
    data: pd.DataFrame,
    response: str,
    factor_a: str,
    factor_b: str,
    levels_a: Sequence[Any],
    levels_b: Sequence[Any],
    alpha: float,
) -> pd.DataFrame:
    """Run pairwise Welch t-tests within each level of the conditioning factor."""
    rows: list[dict[str, Any]] = []
    specs: list[tuple[str, Sequence[Any], str, Sequence[Any]]] = [
        (factor_b, levels_b, factor_a, levels_a),
        (factor_a, levels_a, factor_b, levels_b),
    ]
    for within_col, within_levels, compare_col, compare_levels in specs:
        for within_val in within_levels:
            subset = data[data[within_col].eq(within_val)]
            for lvl_1, lvl_2 in combinations(compare_levels, 2):
                arr_1 = subset.loc[
                    subset[compare_col].eq(lvl_1), response
                ].to_numpy(dtype="float64")
                arr_2 = subset.loc[
                    subset[compare_col].eq(lvl_2), response
                ].to_numpy(dtype="float64")
                t_stat, p_val = stats.ttest_ind(arr_1, arr_2, equal_var=False)
                mean_1 = float(np.mean(arr_1))
                mean_2 = float(np.mean(arr_2))
                p_float = float(p_val)
                rows.append(
                    {
                        "within_factor": within_col,
                        "within_level": within_val,
                        "group_1": str(lvl_1),
                        "group_2": str(lvl_2),
                        "mean_1": mean_1,
                        "mean_2": mean_2,
                        "diff": mean_1 - mean_2,
                        "welch_t": float(t_stat),
                        "p_value": p_float,
                        "reject_h0": p_float < alpha,
                    }
                )
    return pd.DataFrame(rows)


def run_two_way_anova(
    data: pd.DataFrame,
    response: str,
    factor_a: str,
    factor_b: str,
    *,
    alpha: float = 0.05,
    ss_type: Literal["II", "III"] = "III",
) -> TwoWayAnovaResult:
    """Run a two-way factorial ANOVA with assumption checks and post-hoc tests."""
    if not 0 < alpha < 1:
        msg = f"alpha must lie strictly in (0, 1), got {alpha}."
        raise ValueError(msg)
    if ss_type not in {"II", "III"}:
        msg = f"ss_type must be 'II' or 'III', got {ss_type!r}."
        raise ValueError(msg)

    required = [response, factor_a, factor_b]
    missing_cols = [col for col in required if col not in data.columns]
    if missing_cols:
        msg = f"Missing required columns: {missing_cols}."
        raise ValueError(msg)
    if data[required].isna().any().any():
        msg = "Input data contains missing values in response or factor columns."
        raise ValueError(msg)

    levels_a, levels_b = _check_factorial_cells(data, factor_a, factor_b)
    y = data[response].to_numpy(dtype="float64")

    anova_table, residuals = _compute_anova_table(
        y, data, factor_a, factor_b, levels_a, levels_b, alpha, ss_type
    )
    cell_summary, cell_normality, cell_labels, cell_arrays = _compute_cell_tables(
        data, response, factor_a, factor_b, levels_a, levels_b, alpha
    )

    levene_stat, levene_p = stats.levene(*cell_arrays, center="median")
    min_std = float(cell_summary["std"].min())
    sd_ratio = float(cell_summary["std"].max() / min_std) if min_std > 0 else np.inf

    if len(residuals) >= MIN_SHAPIRO_N:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        sh_stat_f, sh_p_f = float(shapiro_stat), float(shapiro_p)
    else:
        sh_stat_f, sh_p_f = np.nan, np.nan

    residual_row = anova_table.loc[anova_table["source"] == "Residual"].iloc[0]
    df_err = int(residual_row["df"])
    mse = float(residual_row["ms"])

    tukey_table = _compute_tukey_table(cell_labels, cell_arrays, alpha)
    lsd_table = _compute_lsd_table(cell_labels, cell_arrays, mse, df_err, alpha)
    simple_effects = _compute_simple_effects(
        data, response, factor_a, factor_b, levels_a, levels_b, alpha
    )

    return TwoWayAnovaResult(
        response=response,
        factor_a=factor_a,
        factor_b=factor_b,
        alpha=alpha,
        ss_type=ss_type,
        n_obs=len(data),
        cell_summary=cell_summary,
        anova_table=anova_table,
        levene_stat=float(levene_stat),
        levene_p=float(levene_p),
        sd_ratio=sd_ratio,
        shapiro_residuals_stat=sh_stat_f,
        shapiro_residuals_p=sh_p_f,
        cell_normality=cell_normality,
        tukey_table=tukey_table,
        lsd_table=lsd_table,
        simple_effects=simple_effects,
        residuals=pd.Series(residuals, index=data.index, name="residuals"),
    )


def _format_p(p_val: float) -> str:
    """Format a p-value consistently for statistical conclusions."""
    return "< 0.001" if p_val < 0.001 else f"= {p_val:.4f}"


def conclude_two_way_anova(
    result: TwoWayAnovaResult,
    hypothesis: TwoWayHypothesis | None = None,
) -> list[str]:
    """Generate statistical conclusions dynamically from a TwoWayAnovaResult."""
    lines: list[str] = []
    alpha = result.alpha
    df_err = int(
        result.anova_table.loc[result.anova_table["source"] == "Residual", "df"].iloc[0]
    )

    var_status = (
        "homogeneity violated" if result.levene_p < alpha else "homogeneity holds"
    )
    norm_status = (
        "normality rejected"
        if (
            not np.isnan(result.shapiro_residuals_p)
            and result.shapiro_residuals_p < alpha
        )
        else "normality holds"
    )
    diag_line = (
        f"Diagnostics (n = {result.n_obs}, df_err = {df_err}): "
        f"Levene W = {result.levene_stat:.3f} (p {_format_p(result.levene_p)}; "
        f"{var_status}), SD ratio = {result.sd_ratio:.2f}; "
        f"Shapiro-Wilk residuals W = {result.shapiro_residuals_stat:.4f} "
        f"(p {_format_p(result.shapiro_residuals_p)}; {norm_status})."
    )
    if result.levene_p < alpha and result.sd_ratio > SD_RATIO_WARN_THRESHOLD:
        diag_line += (
            " Variance heterogeneity is substantial; interpret classical F-tests "
            "alongside a log-transformed response and Welch simple-effects tests."
        )
    lines.append(diag_line)

    interaction_src = f"{result.factor_a}:{result.factor_b}"
    term_labels = [
        (interaction_src, "Interaction", hypothesis.null_ab if hypothesis else "H0ab"),
        (result.factor_a, "Factor A", hypothesis.null_a if hypothesis else "H0a"),
        (result.factor_b, "Factor B", hypothesis.null_b if hypothesis else "H0b"),
    ]
    for source, role, h0_desc in term_labels:
        row = result.anova_table.loc[result.anova_table["source"] == source].iloc[0]
        rejected = bool(row["reject_h0"])
        h0_code = h0_desc.split(":")[0]
        decision = (
            f"Reject H0 ({h0_code})" if rejected else f"Fail to reject H0 ({h0_code})"
        )
        crit_rel = ">" if rejected else "<="
        lines.append(
            f"{role} ({source}, Type {result.ss_type}): {decision} at alpha = "
            f"{alpha:g} -- F({int(row['df'])}, {df_err}) = {row['f_stat']:.3f} "
            f"{crit_rel} f_crit = {row['f_crit']:.3f}, "
            f"p {_format_p(float(row['p_value']))}, "
            f"partial eta^2 = {row['partial_eta_sq']:.4f}."
        )

    sig_lsd = result.lsd_table[result.lsd_table["reject_h0"]]
    nonsig_lsd = result.lsd_table[~result.lsd_table["reject_h0"]]
    lsd_sig_pairs = ", ".join(
        f"{r.group_1} vs {r.group_2} (|diff|={r.abs_diff:.3f} > LSD={r.lsd_crit:.3f})"
        for r in sig_lsd.itertuples()
    )
    lsd_nonsig_pairs = ", ".join(
        f"{r.group_1} vs {r.group_2} (|diff|={r.abs_diff:.3f} <= LSD={r.lsd_crit:.3f})"
        for r in nonsig_lsd.itertuples()
    )
    lines.append(
        f"Fisher's LSD ({len(sig_lsd)}/{len(result.lsd_table)} pairs significant): "
        f"significant = [{lsd_sig_pairs or 'none'}]; "
        f"non-significant = [{lsd_nonsig_pairs or 'none'}]."
    )

    sig_tukey = result.tukey_table[result.tukey_table["reject_h0"]]
    nonsig_tukey = result.tukey_table[~result.tukey_table["reject_h0"]]
    sig_pairs = ", ".join(
        f"{r.group_1} vs {r.group_2} (diff={r.diff:+.3f}, p {_format_p(r.p_value)})"
        for r in sig_tukey.itertuples()
    )
    nonsig_pairs = ", ".join(
        f"{r.group_1} vs {r.group_2} (diff={r.diff:+.3f}, p {_format_p(r.p_value)})"
        for r in nonsig_tukey.itertuples()
    )
    lines.append(
        f"Tukey HSD ({len(sig_tukey)}/{len(result.tukey_table)} pairs significant): "
        f"significant = [{sig_pairs or 'none'}]; "
        f"non-significant = [{nonsig_pairs or 'none'}]."
    )

    simple_summary = "; ".join(
        f"within {r.within_factor}={r.within_level}: {r.group_1} vs {r.group_2} "
        f"(diff={r.diff:+.3f}, Welch t={r.welch_t:.3f}, p {_format_p(r.p_value)}, "
        f"{'significant' if r.reject_h0 else 'not significant'})"
        for r in result.simple_effects.itertuples()
    )
    lines.append(f"Simple effects (Welch t-tests): {simple_summary}.")
    return lines
