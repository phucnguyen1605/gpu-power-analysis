import numpy as np
import pandas as pd
import pytest
from scipy import stats

from gpu_power_analysis import anova, preprocess


def _closed_form_balanced_anova(
    data: pd.DataFrame, response: str, factor_a: str, factor_b: str
) -> dict[str, float]:
    y = data[response].to_numpy(dtype="float64")
    grand_mean = float(np.mean(y))
    n_total = len(y)
    cell_stats = data.groupby([factor_a, factor_b], observed=True)[response].agg(
        ["count", "mean"]
    )
    a_stats = data.groupby(factor_a, observed=True)[response].agg(["count", "mean"])
    b_stats = data.groupby(factor_b, observed=True)[response].agg(["count", "mean"])

    ss_total = float(np.sum((y - grand_mean) ** 2))
    ss_a = float(np.sum(a_stats["count"] * (a_stats["mean"] - grand_mean) ** 2))
    ss_b = float(np.sum(b_stats["count"] * (b_stats["mean"] - grand_mean) ** 2))
    ss_cells = float(
        np.sum(cell_stats["count"] * (cell_stats["mean"] - grand_mean) ** 2)
    )
    ss_ab = ss_cells - ss_a - ss_b
    ss_err = ss_total - ss_cells
    df_err = n_total - len(cell_stats)
    ms_err = ss_err / df_err
    return {
        "ss_a": ss_a,
        "ss_b": ss_b,
        "ss_ab": ss_ab,
        "ss_err": ss_err,
        "ss_total": ss_total,
        "f_a": (ss_a / (len(a_stats) - 1)) / ms_err,
        "f_b": (ss_b / (len(b_stats) - 1)) / ms_err,
        "f_ab": (ss_ab / ((len(a_stats) - 1) * (len(b_stats) - 1))) / ms_err,
    }


def test_balanced_two_by_three_matches_closed_form():
    rows = []
    values = {
        ("A1", "B1"): [10.0, 12.0, 11.0, 13.0],
        ("A1", "B2"): [20.0, 22.0, 19.0, 21.0],
        ("A1", "B3"): [30.0, 29.0, 31.0, 30.0],
        ("A2", "B1"): [16.0, 18.0, 17.0, 15.0],
        ("A2", "B2"): [25.0, 24.0, 26.0, 25.0],
        ("A2", "B3"): [28.0, 27.0, 29.0, 28.0],
    }
    for (a_lvl, b_lvl), ys in values.items():
        for val in ys:
            rows.append({"fa": a_lvl, "fb": b_lvl, "y": val})
    df = pd.DataFrame(rows)

    expected = _closed_form_balanced_anova(df, "y", "fa", "fb")
    res_iii = anova.run_two_way_anova(df, "y", "fa", "fb", ss_type="III")
    res_ii = anova.run_two_way_anova(df, "y", "fa", "fb", ss_type="II")

    tbl = res_iii.anova_table.set_index("source")
    assert tbl.at["fa", "ss"] == pytest.approx(expected["ss_a"], rel=1e-10)
    assert tbl.at["fb", "ss"] == pytest.approx(expected["ss_b"], rel=1e-10)
    assert tbl.at["fa:fb", "ss"] == pytest.approx(expected["ss_ab"], rel=1e-10)
    assert tbl.at["Residual", "ss"] == pytest.approx(expected["ss_err"], rel=1e-10)
    assert tbl.at["Total", "ss"] == pytest.approx(expected["ss_total"], rel=1e-10)
    assert tbl.at["fa", "f_stat"] == pytest.approx(expected["f_a"], rel=1e-10)
    assert tbl.at["fb", "f_stat"] == pytest.approx(expected["f_b"], rel=1e-10)
    assert tbl.at["fa:fb", "f_stat"] == pytest.approx(expected["f_ab"], rel=1e-10)

    sum_parts = (
        tbl.at["fa", "ss"]
        + tbl.at["fb", "ss"]
        + tbl.at["fa:fb", "ss"]
        + tbl.at["Residual", "ss"]
    )
    assert sum_parts == pytest.approx(tbl.at["Total", "ss"], rel=1e-10)

    pd.testing.assert_frame_equal(
        res_iii.anova_table, res_ii.anova_table, atol=1e-10, rtol=1e-10
    )


def test_real_gpu_unbalanced_type_iii_matches_harmonic_mean_formula():
    cleaned = preprocess.load_analysis()
    raw_sample, raw_col = anova.prepare_two_way_sample(
        cleaned,
        "max_power_w",
        "manufacturer",
        "segment",
        include_a=("Nvidia", "AMD"),
    )
    raw_res = anova.run_two_way_anova(
        raw_sample, raw_col, "manufacturer", "segment", alpha=0.05, ss_type="III"
    )

    assert raw_res.n_obs == 1980
    assert len(raw_res.tukey_table) == 6
    assert len(raw_res.lsd_table) == 6
    assert int(raw_res.lsd_table["reject_h0"].sum()) == 4
    cell_counts = {
        (row.manufacturer, row.segment): int(row.count)
        for row in raw_res.cell_summary.itertuples()
    }
    assert cell_counts == {
        ("AMD", "Desktop"): 677,
        ("AMD", "Laptop"): 117,
        ("Nvidia", "Desktop"): 1032,
        ("Nvidia", "Laptop"): 154,
    }

    means = {
        (row.manufacturer, row.segment): float(row.mean)
        for row in raw_res.cell_summary.itertuples()
    }
    n_h = 4.0 / sum(1.0 / cnt for cnt in cell_counts.values())
    y00, y01 = means[("AMD", "Desktop")], means[("AMD", "Laptop")]
    y10, y11 = means[("Nvidia", "Desktop")], means[("Nvidia", "Laptop")]
    ss_a_cf = n_h / 4.0 * ((y00 + y01) - (y10 + y11)) ** 2
    ss_b_cf = n_h / 4.0 * ((y00 + y10) - (y01 + y11)) ** 2
    ss_ab_cf = n_h / 4.0 * ((y00 - y01) - (y10 - y11)) ** 2

    raw_tbl = raw_res.anova_table.set_index("source")
    assert raw_tbl.at["manufacturer", "ss"] == pytest.approx(ss_a_cf, rel=1e-9)
    assert raw_tbl.at["segment", "ss"] == pytest.approx(ss_b_cf, rel=1e-9)
    assert raw_tbl.at["manufacturer:segment", "ss"] == pytest.approx(
        ss_ab_cf, rel=1e-9
    )
    assert bool(raw_tbl.at["manufacturer", "reject_h0"]) is False
    assert bool(raw_tbl.at["segment", "reject_h0"]) is True
    assert bool(raw_tbl.at["manufacturer:segment", "reject_h0"]) is False

    log_sample, log_col = anova.prepare_two_way_sample(
        cleaned,
        "max_power_w",
        "manufacturer",
        "segment",
        include_a=("Nvidia", "AMD"),
        log_transform=True,
    )
    log_res = anova.run_two_way_anova(
        log_sample, log_col, "manufacturer", "segment", alpha=0.05, ss_type="III"
    )
    log_tbl = log_res.anova_table.set_index("source")
    assert bool(log_tbl.at["manufacturer", "reject_h0"]) is True
    assert bool(log_tbl.at["segment", "reject_h0"]) is True
    assert bool(log_tbl.at["manufacturer:segment", "reject_h0"]) is True
    assert log_res.sd_ratio < raw_res.sd_ratio
    assert len(log_res.lsd_table) == 6
    assert int(log_res.lsd_table["reject_h0"].sum()) == 6


def test_fisher_lsd_matches_unequal_n_slide_formula():
    cleaned = preprocess.load_analysis()
    for log_transform in (False, True):
        sample, col = anova.prepare_two_way_sample(
            cleaned,
            "max_power_w",
            "manufacturer",
            "segment",
            include_a=("Nvidia", "AMD"),
            log_transform=log_transform,
        )
        res = anova.run_two_way_anova(
            sample, col, "manufacturer", "segment", alpha=0.05, ss_type="III"
        )
        residual_row = res.anova_table.loc[
            res.anova_table["source"] == "Residual"
        ].iloc[0]
        df_err = int(residual_row["df"])
        mse = float(residual_row["ms"])
        expected_t_crit = float(stats.t.ppf(0.975, df_err))
        counts = {
            f"{row.manufacturer}-{row.segment}": int(row.count)
            for row in res.cell_summary.itertuples()
        }
        for row in res.lsd_table.itertuples():
            n1 = counts[row.group_1]
            n2 = counts[row.group_2]
            expected_lsd = expected_t_crit * float(np.sqrt(mse * (1.0 / n1 + 1.0 / n2)))
            assert row.t_crit == pytest.approx(expected_t_crit, rel=1e-12)
            assert row.lsd_crit == pytest.approx(expected_lsd, rel=1e-12)
            assert bool(row.reject_h0) is bool(row.abs_diff > row.lsd_crit)


def test_prepare_and_run_validation_errors():
    cleaned = preprocess.load_analysis()
    with pytest.raises(ValueError, match="ATI, Laptop"):
        anova.prepare_two_way_sample(
            cleaned, "max_power_w", "manufacturer", "segment"
        )

    bad_y = pd.DataFrame(
        {
            "manufacturer": [
                "AMD",
                "AMD",
                "AMD",
                "AMD",
                "Nvidia",
                "Nvidia",
                "Nvidia",
                "Nvidia",
            ],
            "segment": [
                "Desktop",
                "Desktop",
                "Laptop",
                "Laptop",
                "Desktop",
                "Desktop",
                "Laptop",
                "Laptop",
            ],
            "max_power_w": [100.0, 0.0, 30.0, 40.0, 120.0, 130.0, 50.0, 60.0],
        }
    )
    with pytest.raises(ValueError, match="non-positive"):
        anova.prepare_two_way_sample(
            bad_y, "max_power_w", "manufacturer", "segment", log_transform=True
        )

    with pytest.raises(ValueError, match="alpha"):
        anova.run_two_way_anova(
            bad_y, "max_power_w", "manufacturer", "segment", alpha=1.2
        )

    with pytest.raises(ValueError, match="ss_type"):
        anova.run_two_way_anova(
            bad_y,
            "max_power_w",
            "manufacturer",
            "segment",
            ss_type="I",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="Missing required columns"):
        anova.run_two_way_anova(bad_y, "missing_col", "manufacturer", "segment")

    with_na = bad_y.copy()
    with_na.loc[0, "max_power_w"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        anova.run_two_way_anova(with_na, "max_power_w", "manufacturer", "segment")


def test_conclude_two_way_anova_reports_decisions():
    cleaned = preprocess.load_analysis()
    hyp = anova.build_gpu_power_hypotheses()
    raw_sample, raw_col = anova.prepare_two_way_sample(
        cleaned,
        "max_power_w",
        "manufacturer",
        "segment",
        include_a=("Nvidia", "AMD"),
    )
    raw_res = anova.run_two_way_anova(raw_sample, raw_col, "manufacturer", "segment")
    lines = anova.conclude_two_way_anova(raw_res, hyp)

    assert len(lines) == 7
    text = "\n".join(lines)
    assert "Fail to reject H0 (H0ab)" in text
    assert "Fail to reject H0 (H0a)" in text
    assert "Reject H0 (H0b)" in text
    assert "Fisher's LSD" in text
    assert "Variance heterogeneity is substantial" in text
