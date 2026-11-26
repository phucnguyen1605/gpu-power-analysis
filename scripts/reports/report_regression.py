"""Run the GPU multiple regression as a single self-contained Python script.

Place this file at scripts/reports/report_regression.py in the supplied project.
No custom gpu_power_analysis module or companion Python file is imported.

Usage from the project root:
    python scripts/reports/report_regression.py --alpha 0.005
    python report_regression.py --data data/processed/gpus_analysis.csv

External dependencies are numpy, pandas, scipy, statsmodels and matplotlib;
pyarrow is additionally required for Parquet. These are in the latest project.
Outputs go to report/generated/ unless --output-dir is specified. The tables
are LaTeX row bodies, matching report_anova.py, with Reg macros in a separate
regression_numbers.tex file. Coefficient effects are also exported separately.

H0: the four technical slopes are jointly zero, conditional on vendor/segment.
Inference is clustered by the supplied architecture labels (CR1); clusters are
assumed independent. Validation holds out entire architectures. Only rows
missing core clock are excluded, and the shared input is never modified.

Intervals for example configurations cover the geometric conditional mean.
They are not arithmetic-mean confidence intervals or individual prediction
intervals. An overall smearing factor does not fix heteroskedasticity by itself.
Cook flags are diagnostic, not an automatic reason to remove observations.
"""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.regression.linear_model import RegressionResults
from statsmodels.stats.contrast import ContrastResults
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor

mpl.rcParams.update(
    {"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"}
)

TECHNICAL = ["ln_bw", "ln_core_speed", "process_nm", "ln_memory_gb"]
NUMERIC = [
    "max_power_w",
    "memory_bandwidth_gb_s",
    "core_speed_mhz",
    "memory_mb",
    "process_nm",
]
CATEGORICAL = ["manufacturer", "segment", "architecture"]


@dataclass
class Analysis:
    """Hold the sample, design, fitted models and primary joint test."""

    sample: pd.DataFrame
    excluded: pd.DataFrame
    design: pd.DataFrame
    response: pd.Series
    ols: RegressionResults
    hc3: RegressionResults
    clustered: RegressionResults
    primary: ContrastResults
    clusters: int


def prepare_sample(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate required fields and exclude rows missing only the core clock.

    Args:
        data: The shared cleaned analysis dataset.

    Returns:
        A complete regression sample and the rows excluded for missing clock.

    Raises:
        ValueError: If required data are absent, invalid or nonpositive.
    """
    missing = sorted(set(NUMERIC + CATEGORICAL) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    sample = data.copy()
    for column in NUMERIC:
        sample[column] = pd.to_numeric(sample[column], errors="raise").astype(float)
    required = [column for column in NUMERIC if column != "core_speed_mhz"]
    if sample[required + CATEGORICAL].isna().any().any():
        raise ValueError("Unexpected missing values outside core_speed_mhz")
    if np.isinf(sample[NUMERIC].to_numpy()).any():
        raise ValueError("Infinite values in physical specifications")
    excluded = sample.loc[sample["core_speed_mhz"].isna()].copy()
    sample = sample.loc[sample["core_speed_mhz"].notna()].reset_index(drop=True)
    if sample.empty or (sample[NUMERIC] <= 0).any().any():
        raise ValueError("The sample must contain positive physical specifications")
    for column in CATEGORICAL:
        sample[column] = sample[column].astype(str).str.strip()
        if sample[column].eq("").any():
            raise ValueError(f"Empty labels in {column}")
    if "Nvidia" not in set(sample["manufacturer"]):
        raise ValueError("The Nvidia reference manufacturer must be present")
    if set(sample["segment"]) != {"Desktop", "Laptop"}:
        raise ValueError("The model requires Desktop and Laptop labels")
    for source, target in [
        ("max_power_w", "ln_power"),
        ("memory_bandwidth_gb_s", "ln_bw"),
        ("core_speed_mhz", "ln_core_speed"),
    ]:
        sample[target] = np.log(sample[source])
    sample["ln_memory_gb"] = np.log(sample["memory_mb"] / 1024.0)
    return sample, excluded


def fit_model(data: pd.DataFrame) -> Analysis:
    """Fit OLS and robust covariance variants, then test four technical slopes.

    Args:
        data: The shared cleaned GPU dataset.

    Returns:
        The fitted analysis with an architecture-clustered primary Wald test.

    Raises:
        ValueError: If the design is singular or there are too few clusters.
    """
    sample, excluded = prepare_sample(data)
    design = sm.add_constant(sample[TECHNICAL], has_constant="add")
    for manufacturer in sorted(set(sample["manufacturer"]) - {"Nvidia"}):
        design[f"mfg_{manufacturer}"] = sample["manufacturer"].eq(manufacturer)
    design["is_notebook"] = sample["segment"].eq("Laptop")
    design = design.astype(float)
    response = sample["ln_power"].astype(float)
    if (
        len(sample) <= design.shape[1]
        or np.linalg.matrix_rank(design) < design.shape[1]
    ):
        raise ValueError("Insufficient degrees of freedom or a singular design")
    groups, labels = pd.factorize(sample["architecture"])
    if len(labels) <= design.shape[1]:
        raise ValueError("Too few independent architecture clusters for inference")
    ols = sm.OLS(response, design).fit()
    hc3 = ols.get_robustcov_results(cov_type="HC3", use_t=True)
    clustered = ols.get_robustcov_results(
        cov_type="cluster",
        groups=groups,
        use_correction=True,
        df_correction=True,
        use_t=True,
    )
    restriction = np.zeros((4, design.shape[1]))
    for index, term in enumerate(TECHNICAL):
        restriction[index, design.columns.get_loc(term)] = 1.0
    return Analysis(
        sample,
        excluded,
        design,
        response,
        ols,
        hc3,
        clustered,
        clustered.f_test(restriction),
        len(labels),
    )


def coefficient_table(analysis: Analysis, alpha: float) -> pd.DataFrame:
    """Return coefficients, covariance comparisons and pointwise cluster CIs."""
    interval = analysis.clustered.conf_int(alpha=alpha)
    return pd.DataFrame(
        {
            "term": analysis.design.columns,
            "estimate": analysis.clustered.params,
            "se_ols": analysis.ols.bse.to_numpy(),
            "se_hc3": analysis.hc3.bse,
            "se_cluster": analysis.clustered.bse,
            "p_cluster": analysis.clustered.pvalues,
            "ci_low": interval[:, 0],
            "ci_high": interval[:, 1],
        }
    )


def diagnostics(analysis: Analysis) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return VIF and conventional diagnostic tests, without deleting points."""
    design = analysis.design.to_numpy()
    vif = pd.DataFrame(
        {
            "term": analysis.design.columns[1:],
            "vif": [
                variance_inflation_factor(design, i) for i in range(1, design.shape[1])
            ],
        }
    )
    residual = analysis.ols.resid
    bp = het_breuschpagan(residual, design)
    # White's auxiliary design contains redundant dummy squares/interactions.
    # The diagnostic uses a pseudoinverse and rank-based degrees of freedom.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"The design matrix is rank-deficient\..*"
        )
        white = het_white(residual, design)
    shapiro = stats.shapiro(residual)
    tests = pd.DataFrame(
        [
            {"test": "Breusch-Pagan LM", "statistic": bp[0], "p": bp[1]},
            {"test": "White LM", "statistic": white[0], "p": white[1]},
            {
                "test": "Shapiro-Wilk",
                "statistic": shapiro.statistic,
                "p": shapiro.pvalue,
            },
        ]
    )
    return vif, tests


def architecture_folds(
    groups: pd.Series, folds: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Balance folds by size while keeping complete architecture labels together.

    Args:
        groups: Architecture labels in row order; no response values are used.
        folds: Number of held-out folds.

    Returns:
        Training and validation row positions for each fold.

    Raises:
        ValueError: If labels are missing or there are too few groups.
    """
    if groups.isna().any() or folds < 2 or groups.nunique() < folds:
        raise ValueError(
            "Cross-validation requires enough complete architecture labels"
        )
    counts = groups.value_counts()
    sizes = np.zeros(folds, dtype=int)
    assignment = {}
    for label in sorted(counts.index, key=lambda item: (-counts[item], str(item))):
        fold = int(sizes.argmin())
        assignment[label] = fold
        sizes[fold] += int(counts[label])
    allocation = groups.map(assignment).to_numpy()
    return [
        (np.flatnonzero(allocation != fold), np.flatnonzero(allocation == fold))
        for fold in range(folds)
    ]


def cross_validate(
    analysis: Analysis,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Evaluate held-out architectures with training-only smearing corrections.

    Returns:
        Per-row predictions, fold membership checks and aggregate Watt errors.
        Smearing is an overall correction, not a guaranteed conditional mean.
    """
    predictions = np.zeros(len(analysis.sample))
    naive = np.zeros(len(analysis.sample))
    fold_ids = np.zeros(len(analysis.sample), dtype=int)
    fold_rows = []
    groups = analysis.sample["architecture"]
    for fold, (train, test) in enumerate(architecture_folds(groups), start=1):
        train_design = analysis.design.iloc[train]
        if np.linalg.matrix_rank(train_design) < train_design.shape[1]:
            raise ValueError("A training fold cannot identify all model coefficients")
        fitted = sm.OLS(analysis.response.iloc[train], train_design).fit()
        naive[test] = np.exp(fitted.predict(analysis.design.iloc[test]))
        predictions[test] = naive[test] * np.exp(fitted.resid).mean()
        fold_ids[test] = fold
        overlap = set(groups.iloc[train]) & set(groups.iloc[test])
        if overlap:
            raise ValueError("Architecture leakage across training and validation")
        fold_rows.append(
            {
                "fold": fold,
                "training_rows": len(train),
                "validation_rows": len(test),
                "architecture_overlap": len(overlap),
            }
        )
    actual = analysis.sample["max_power_w"].to_numpy()
    row_table = analysis.sample[["gpu_id", "name", "architecture"]].copy()
    row_table["fold"] = fold_ids
    row_table["actual_w"] = actual
    row_table["predicted_w_smearing"] = predictions
    row_table["predicted_w_naive"] = naive
    metrics = {
        "rmse": float(np.sqrt(np.mean((actual - predictions) ** 2))),
        "mae": float(np.mean(np.abs(actual - predictions))),
        "r2": float(
            1
            - np.sum((actual - predictions) ** 2)
            / np.sum((actual - actual.mean()) ** 2)
        ),
    }
    return row_table, pd.DataFrame(fold_rows), metrics


def illustrative_estimates(analysis: Analysis, alpha: float) -> pd.DataFrame:
    """Estimate three observed configurations with geometric-mean cluster CIs.

    These are in-sample illustrations. No arithmetic-mean CI or prediction
    interval is claimed under the observed nonconstant residual variance.
    """
    order = np.argsort(analysis.sample["memory_bandwidth_gb_s"].to_numpy())
    indices = order[[len(order) // 4, len(order) // 2, 3 * len(order) // 4]]
    design = analysis.design.iloc[indices].to_numpy()
    predicted_log = design @ analysis.clustered.params
    variance = np.einsum("ij,jk,ik->i", design, analysis.clustered.cov_params(), design)
    critical = stats.t.ppf(1 - alpha / 2, analysis.clusters - 1)
    margin = critical * np.sqrt(np.maximum(variance, 0))
    table = analysis.sample.iloc[indices][["gpu_id", "name", *NUMERIC]].copy()
    table["geometric_mean_w"] = np.exp(predicted_log)
    table["geometric_ci_low"] = np.exp(predicted_log - margin)
    table["geometric_ci_high"] = np.exp(predicted_log + margin)
    table["smearing_adjusted_w"] = (
        np.exp(predicted_log) * np.exp(analysis.ols.resid).mean()
    )
    return table


logger = logging.getLogger("report_regression")


def decimal(value: float, digits: int = 3) -> str:
    """Format a decimal comma for use in LaTeX math."""
    return f"{value:.{digits}f}".replace(".", "{,}")


def fmt_pval_tex(value: float) -> str:
    """Format p-values safely inside and outside LaTeX math mode.

    Args:
        value: A p-value between zero and one.

    Returns:
        A LaTeX ensuremath expression without literal dollar delimiters.

    Raises:
        ValueError: If the supplied probability is invalid.
    """
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("A p-value must be finite and between zero and one")
    if value == 0:
        content = r"< 10^{-300}"
    elif value < 0.001:
        mantissa, exponent = f"{value:.2e}".split("e")
        content = rf"{mantissa.replace('.', '{,}')} \cdot 10^{{{int(exponent)}}}"
    else:
        content = decimal(value, 4)
    return rf"\ensuremath{{{content}}}"


def latex_escape(text: str) -> str:
    """Escape special LaTeX characters in a plain-text dynamic label."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def add_effects(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Add exact geometric-mean effects, distinguishing numeric and dummy terms.

    Args:
        coefficients: The fitted coefficient summary.

    Returns:
        A copy with percentage effects and the comparison defining each effect.
    """
    result = coefficients.copy()
    effects = []
    bases = []
    for row in result.itertuples():
        if row.term in {"ln_bw", "ln_core_speed", "ln_memory_gb"}:
            effects.append(100 * np.expm1(row.estimate * np.log1p(0.01)))
            bases.append("Predictor +1 percent; other predictors fixed")
        elif row.term == "process_nm":
            effects.append(100 * np.expm1(row.estimate))
            bases.append("Process size +1 nm; other predictors fixed")
        elif row.term == "is_notebook" or row.term.startswith("mfg_"):
            effects.append(100 * np.expm1(row.estimate))
            bases.append("Category versus reference; other predictors fixed")
        else:
            effects.append(np.nan)
            bases.append("Intercept: no percentage-change interpretation")
    result["effect_percent"] = effects
    result["effect_basis"] = bases
    return result


def cook_contour(leverage: np.ndarray, parameters: int, cutoff: float) -> np.ndarray:
    """Return Cook contour bounds for internally studentized residuals.

    Args:
        leverage: Leverage values strictly between zero and one.
        parameters: Number of model parameters, including the intercept.
        cutoff: The Cook distance defining the contour.

    Returns:
        Positive residual bounds; negative bounds have the opposite sign.

    Raises:
        ValueError: If leverage, parameter count or cutoff are invalid.
    """
    leverage = np.asarray(leverage, dtype=float)
    if (
        not np.isfinite(leverage).all()
        or (leverage <= 0).any()
        or (leverage >= 1).any()
        or parameters <= 0
        or not np.isfinite(cutoff)
        or cutoff < 0
    ):
        raise ValueError("Invalid leverage, parameter count or Cook cutoff")
    return np.sqrt(cutoff * parameters * (1 - leverage) / leverage)


def project_root() -> Path:
    """Find the data directory from the current folder or script ancestors."""
    current = Path.cwd()
    for candidate in [current, *Path(__file__).resolve().parents]:
        if (candidate / "data" / "processed").is_dir():
            return candidate
    return current


def plot_diagnostics(analysis: Analysis, output: Path) -> None:
    """Write four residual diagnostics, using the correct Cook contour scale."""
    influence = analysis.ols.get_influence()
    fitted = np.asarray(analysis.ols.fittedvalues)
    residual = np.asarray(analysis.ols.resid)
    studentized = influence.resid_studentized_internal
    leverage = influence.hat_matrix_diag
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].scatter(fitted, residual, s=9, alpha=0.35)
    smooth = sm.nonparametric.lowess(residual, fitted, frac=0.3)
    axes[0, 0].plot(smooth[:, 0], smooth[:, 1], color="darkorange")
    axes[0, 0].axhline(0, color="black", linewidth=0.7)
    axes[0, 0].set(
        title="Residuals vs Fitted", xlabel="Fitted ln(P)", ylabel="Residual"
    )
    stats.probplot(studentized, plot=axes[0, 1])
    axes[0, 1].set(title="Normal Q-Q", ylabel="Studentized residual")
    scale = np.sqrt(np.abs(studentized))
    axes[1, 0].scatter(fitted, scale, s=9, alpha=0.35)
    smooth = sm.nonparametric.lowess(scale, fitted, frac=0.3)
    axes[1, 0].plot(smooth[:, 0], smooth[:, 1], color="darkorange")
    axes[1, 0].set(
        title="Scale-Location", xlabel="Fitted ln(P)", ylabel="sqrt(|residual|)"
    )
    axes[1, 1].scatter(leverage, studentized, s=9, alpha=0.35)
    grid = np.linspace(
        max(1e-5, leverage.min()), min(leverage.max() * 1.05, 1 - 1e-10), 150
    )
    bound = cook_contour(grid, analysis.design.shape[1], 4 / len(fitted))
    axes[1, 1].plot(grid, bound, "r--", linewidth=0.8, label="Cook D = 4/n")
    axes[1, 1].plot(grid, -bound, "r--", linewidth=0.8)
    axes[1, 1].set(
        title="Residuals vs Leverage", xlabel="Leverage", ylabel="Studentized residual"
    )
    axes[1, 1].legend(fontsize=8)
    fig.tight_layout()
    for extension in ["pdf", "png"]:
        fig.savefig(output / f"regression_diagnostics.{extension}", dpi=180)
    plt.close(fig)


def plot_validation(predictions: pd.DataFrame, output: Path) -> None:
    """Plot held-out predictions from architecture-grouped cross-validation."""
    actual = predictions["actual_w"]
    predicted = predictions["predicted_w_smearing"]
    limit = max(actual.max(), predicted.max()) * 1.05
    fig, ax = plt.subplots(figsize=(5.7, 4.5))
    ax.scatter(actual, predicted, s=10, alpha=0.35)
    ax.plot([0, limit], [0, limit], "r--", linewidth=1)
    ax.set(
        xlabel="Recorded power (W)",
        ylabel="Held-out prediction (W)",
        title="Five-fold validation by architecture",
        xlim=(0, limit),
        ylim=(0, limit),
    )
    fig.tight_layout()
    for extension in ["pdf", "png"]:
        fig.savefig(output / f"regression_actual_vs_pred.{extension}", dpi=180)
    plt.close(fig)


def export_tables(
    analysis: Analysis,
    coefficients: pd.DataFrame,
    vif: pd.DataFrame,
    tests: pd.DataFrame,
    estimates: pd.DataFrame,
    metrics: dict[str, float],
    alpha: float,
    output: Path,
) -> None:
    """Write LaTeX table bodies to match the ANOVA report integration style."""
    header = "% Generated by scripts/reports/report_regression.py; do not edit.\n"
    names = {
        "const": r"$\beta_0$",
        "ln_bw": r"$\ln B$",
        "ln_core_speed": r"$\ln F$",
        "process_nm": r"$L$ (nm)",
        "ln_memory_gb": r"$\ln M$",
        "mfg_AMD": "AMD vs Nvidia",
        "mfg_ATI": "ATI vs Nvidia",
        "is_notebook": "Laptop vs Desktop",
    }
    lines = []
    for row in coefficients.itertuples():
        name = names.get(row.term, latex_escape(row.term))
        interval = f"$[{decimal(row.ci_low)}; {decimal(row.ci_high)}]$"
        lines.append(
            f"{name} & {decimal(row.estimate)} & {decimal(row.se_ols)} & "
            f"{decimal(row.se_hc3)} & {decimal(row.se_cluster)} & {interval} & "
            f"${fmt_pval_tex(row.p_cluster)}$ " + r"\\"
        )
    (output / "regression_coef_table.tex").write_text(header + "\n".join(lines) + "\n")
    lines = []
    for row in coefficients.itertuples():
        label = names.get(row.term, latex_escape(row.term))
        effect = "---" if np.isnan(row.effect_percent) else decimal(row.effect_percent)
        lines.append(f"{label} & {effect} & {latex_escape(row.effect_basis)} " + r"\\")
    (output / "regression_effects_table.tex").write_text(
        header + "\n".join(lines) + "\n"
    )
    lines = [
        f"{names.get(row.term, latex_escape(row.term))} & {decimal(row.vif, 2)} "
        + r"\\"
        for row in vif.itertuples()
    ]
    (output / "regression_vif_table.tex").write_text(header + "\n".join(lines) + "\n")
    lines = [
        f"{row.test} & {decimal(row.statistic)} & ${fmt_pval_tex(row.p)}$ " + r"\\"
        for row in tests.itertuples()
    ]
    (output / "regression_diagnostics_table.tex").write_text(
        header + "\n".join(lines) + "\n"
    )
    lines = [
        f"{int(row.gpu_id)} & {decimal(row.geometric_mean_w, 1)} & "
        f"$[{decimal(row.geometric_ci_low, 1)}; "
        f"{decimal(row.geometric_ci_high, 1)}]$ & "
        f"{decimal(row.smearing_adjusted_w, 1)} " + r"\\"
        for row in estimates.itertuples()
    ]
    (output / "regression_forecast_table.tex").write_text(
        header + "\n".join(lines) + "\n"
    )
    primary_p = float(np.asarray(analysis.primary.pvalue).item())
    values = {
        "N": len(analysis.sample),
        "RowsExcluded": len(analysis.excluded),
        "RowsInput": len(analysis.sample) + len(analysis.excluded),
        "Clusters": analysis.clusters,
        "Alpha": decimal(alpha, 3),
        "Confidence": decimal(100 * (1 - alpha), 1),
        "RSquared": decimal(analysis.ols.rsquared, 4),
        "AdjRSquared": decimal(analysis.ols.rsquared_adj, 4),
        "PrimaryF": decimal(float(np.asarray(analysis.primary.fvalue).item()), 3),
        "PrimaryP": fmt_pval_tex(primary_p),
        "PrimaryDfOne": int(analysis.primary.df_num),
        "PrimaryDfTwo": int(analysis.primary.df_denom),
        "MaxVIF": decimal(vif["vif"].max(), 3),
        "CVRMSE": decimal(metrics["rmse"], 2),
        "CVMAE": decimal(metrics["mae"], 2),
        "CVRSquaredW": decimal(metrics["r2"], 4),
        "Reject": int(primary_p <= alpha),
    }
    macros = [
        rf"\newcommand{{\Reg{name}}}{{{value}}}" for name, value in values.items()
    ]
    (output / "regression_numbers.tex").write_text(header + "\n".join(macros) + "\n")


def main() -> None:
    """Run the analysis and write reproducible report artifacts."""
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "data" / "processed" / "gpus_analysis.parquet",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "report" / "generated"
    )
    args = parser.parse_args()
    if not 0 < args.alpha < 1:
        parser.error("alpha must be between zero and one")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    source = (
        pd.read_parquet(args.data)
        if args.data.suffix == ".parquet"
        else pd.read_csv(args.data)
    )
    analysis = fit_model(source)
    coefficients = add_effects(coefficient_table(analysis, args.alpha))
    vif, tests = diagnostics(analysis)
    predictions, folds, metrics = cross_validate(analysis)
    estimates = illustrative_estimates(analysis, args.alpha)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    for name, table in {
        "coefficients": coefficients,
        "vif": vif,
        "diagnostic_tests": tests,
        "sample": analysis.sample,
        "excluded": analysis.excluded,
        "cv_predictions": predictions,
        "cv_folds": folds,
        "forecasts": estimates,
    }.items():
        table.to_csv(
            output / f"regression_{name}.csv", index=False, encoding="utf-8-sig"
        )
    result = {
        "hypothesis": (
            "Four technical slopes jointly zero, controlling vendor and segment"
        ),
        "alpha": args.alpha,
        "n": len(analysis.sample),
        "excluded": len(analysis.excluded),
        "clusters": analysis.clusters,
        "covariance": "architecture-clustered CR1",
        "f": float(np.asarray(analysis.primary.fvalue).item()),
        "df_num": float(analysis.primary.df_num),
        "df_denom": float(analysis.primary.df_denom),
        "p": float(np.asarray(analysis.primary.pvalue).item()),
        "r_squared_log": float(analysis.ols.rsquared),
        "grouped_cv_watts": metrics,
        "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scipy", "statsmodels"]
        },
        "notes": [
            "Pointwise coefficient tests are not adjusted for multiple comparisons.",
            "Conventional residual diagnostic p-values are not cluster-corrected.",
            "Clusters use supplied architecture labels and are assumed independent.",
            "Overall smearing does not guarantee the conditional arithmetic mean.",
            "Forecast CIs describe geometric means, not new individual GPU outcomes.",
        ],
    }
    (output / "regression_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (output / "regression_summary.txt").write_text(
        str(analysis.clustered.summary(alpha=args.alpha)), encoding="utf-8"
    )
    export_tables(
        analysis, coefficients, vif, tests, estimates, metrics, args.alpha, output
    )
    plot_diagnostics(analysis, output)
    plot_validation(predictions, output)
    logger.info(
        "Primary joint test: F(%s, %s)=%.6g, p=%.8g",
        result["df_num"],
        result["df_denom"],
        result["f"],
        result["p"],
    )
    logger.info(
        "Rows=%s; excluded=%s; clusters=%s; alpha=%s",
        result["n"],
        result["excluded"],
        result["clusters"],
        args.alpha,
    )
    logger.info("Grouped CV on Watts: %s", metrics)


if __name__ == "__main__":
    main()
