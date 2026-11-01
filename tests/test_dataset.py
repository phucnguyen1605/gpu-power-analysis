import pandas as pd
import pytest

from gpu_power_analysis import clean, dataset, paths


def test_pool_rare_keeps_missing():
    values = pd.Series(["GDDR5"] * 3 + ["HBM-2", None], dtype="category")
    result = dataset.pool_rare(values, min_size=2, other="Other")
    assert result.tolist()[:4] == ["GDDR5", "GDDR5", "GDDR5", "Other"]
    assert pd.isna(result.iloc[4])


@pytest.fixture(scope="module")
def built() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = clean.read_raw(paths.RAW_GPUS)
    base = clean.add_flags(clean.parse_columns(raw), raw)
    rows, counts = dataset.filter_rows(base)
    return base, dataset.derive_variables(rows), counts


def test_pool_rare_counts_only_selected_rows():
    values = pd.Series(["A", "A", "A", "B", "B", "B"], dtype="category")
    counted = pd.Series([True, True, True, True, False, False])
    result = dataset.pool_rare(values, min_size=2, other="Other", counted=counted)
    assert result.tolist() == ["A", "A", "A", "Other", "Other", "Other"]


def test_needs_aux_power_threshold_is_strict(built):
    base, _, _ = built
    rows = base.head(3).copy()
    rows["max_power_w"] = pd.array([75.0, 76.0, None], dtype="Float64")
    derived = dataset.derive_variables(rows.reset_index(drop=True))
    assert derived["needs_aux_power"].tolist() == [False, True, pd.NA]


def test_every_column_has_a_role_and_a_use(built):
    _, analysis, _ = built
    assert list(analysis.columns) == list(dataset.COLUMNS)
    for role, used_in, description in dataset.COLUMNS.values():
        assert role and used_in and description


def test_row_counts_never_increase(built):
    _, analysis, counts = built
    assert counts["rows"].is_monotonic_decreasing
    assert counts["rows"].iloc[-1] == len(analysis)


def test_kept_rows_satisfy_every_filter(built):
    base, analysis, _ = built
    kept = base.set_index("gpu_id").loc[analysis["gpu_id"]]
    assert kept["is_dedicated"].all()
    assert not kept["is_unreleased"].any()
    assert not kept["is_exact_duplicate"].any()


def test_values_are_carried_over_unchanged(built):
    base, analysis, _ = built
    source = base.set_index("gpu_id").loc[analysis["gpu_id"]].reset_index()
    for column in ["max_power_w", "tmus", "memory_bandwidth_gb_s", "release_year"]:
        pd.testing.assert_series_equal(
            analysis[column], source[column], check_names=False
        )
