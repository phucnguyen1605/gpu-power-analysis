"""Run two-way factorial ANOVA on the cleaned GPU analysis dataset.

Usage:
    python scripts/02_anova_two_way.py

Input:
    data/processed/gpus_analysis.parquet
"""

import logging

import numpy as np
import pandas as pd

from gpu_power_analysis import anova, preprocess

logger = logging.getLogger("anova_two_way")

ALPHA = 0.05
FACTORS = ("manufacturer", "segment")
INCLUDED_MANUFACTURERS = ("Nvidia", "AMD")


def _log_hypothesis(title: str, hypothesis: anova.TwoWayHypothesis) -> None:
    logger.info("=== %s (alpha = %.2f) ===", title, hypothesis.alpha)
    logger.info("Question: %s", hypothesis.research_question)
    logger.info("  %s", hypothesis.null_a)
    logger.info("  %s", hypothesis.alt_a)
    logger.info("  %s", hypothesis.null_b)
    logger.info("  %s", hypothesis.alt_b)
    logger.info("  %s", hypothesis.null_ab)
    logger.info("  %s", hypothesis.alt_ab)


def _log_result(
    title: str,
    result: anova.TwoWayAnovaResult,
    hypothesis: anova.TwoWayHypothesis,
    *,
    show_geometric_mean: bool = False,
) -> None:
    logger.info(
        "\n=== %s (Type %s SS, n = %d) ===", title, result.ss_type, result.n_obs
    )

    summary = result.cell_summary.copy()
    if show_geometric_mean:
        summary["geom_mean_w"] = np.exp(summary["mean"])
    logger.info("Cell summary:\n%s", summary.to_string(index=False))

    logger.info(
        "Assumption checks:\n"
        "  Levene (median): W = %.3f, p = %.4g, SD ratio = %.2f\n"
        "  Shapiro-Wilk (residuals): W = %.4f, p = %.4g\n%s",
        result.levene_stat,
        result.levene_p,
        result.sd_ratio,
        result.shapiro_residuals_stat,
        result.shapiro_residuals_p,
        result.cell_normality.to_string(index=False),
    )

    logger.info("ANOVA table:\n%s", result.anova_table.to_string(index=False))
    logger.info(
        "Fisher's LSD pairwise comparisons:\n%s",
        result.lsd_table.to_string(index=False),
    )
    logger.info(
        "Tukey HSD pairwise comparisons:\n%s", result.tukey_table.to_string(index=False)
    )
    logger.info(
        "Simple effects (Welch t-tests):\n%s",
        result.simple_effects.to_string(index=False),
    )

    logger.info("Conclusions:")
    for line in anova.conclude_two_way_anova(result, hypothesis):
        logger.info("  - %s", line)


def main() -> None:
    """Define hypotheses, load cleaned GPU data, run two-way ANOVA, and conclude."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    hyp_raw = anova.build_gpu_power_hypotheses(
        response_label="rated maximum power (W)", alpha=ALPHA
    )
    hyp_log = anova.build_gpu_power_hypotheses(
        response_label="log-transformed rated maximum power ln(W)", alpha=ALPHA
    )

    logger.info("Step 1/4  State research questions and formal hypotheses")
    _log_hypothesis("Raw Scale (max_power_w)", hyp_raw)
    _log_hypothesis("Log Scale (log_max_power_w)", hyp_log)

    logger.info("\nStep 2/4  Load cleaned analysis dataset")
    cleaned = preprocess.load_analysis()
    crosstab = pd.crosstab(cleaned["manufacturer"], cleaned["segment"], margins=True)
    logger.info(
        "Loaded %d rows x %d columns.\nCross-tab (manufacturer x segment):\n%s",
        *cleaned.shape,
        crosstab.to_string(),
    )
    logger.info(
        "Note: ATI has 0 Laptop GPUs, so the factorial design restricts "
        "manufacturer to %s for complete cells.",
        list(INCLUDED_MANUFACTURERS),
    )

    logger.info("\nStep 3/4  Run Two-Way ANOVA on raw max_power_w")
    raw_sample, raw_col = anova.prepare_two_way_sample(
        cleaned,
        "max_power_w",
        FACTORS[0],
        FACTORS[1],
        include_a=INCLUDED_MANUFACTURERS,
        log_transform=False,
    )
    raw_result = anova.run_two_way_anova(
        raw_sample,
        raw_col,
        FACTORS[0],
        FACTORS[1],
        alpha=ALPHA,
        ss_type="III",
    )
    _log_result("Two-Way ANOVA: max_power_w", raw_result, hyp_raw)

    logger.info(
        "\nStep 4/4  Run Two-Way ANOVA on log_max_power_w "
        "(stabilizes right-skewed variance)"
    )
    log_sample, log_col = anova.prepare_two_way_sample(
        cleaned,
        "max_power_w",
        FACTORS[0],
        FACTORS[1],
        include_a=INCLUDED_MANUFACTURERS,
        log_transform=True,
    )
    log_result = anova.run_two_way_anova(
        log_sample,
        log_col,
        FACTORS[0],
        FACTORS[1],
        alpha=ALPHA,
        ss_type="III",
    )
    _log_result(
        "Two-Way ANOVA: log_max_power_w",
        log_result,
        hyp_log,
        show_geometric_mean=True,
    )


if __name__ == "__main__":
    main()
