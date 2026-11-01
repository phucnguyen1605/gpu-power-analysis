import pandas as pd
import pytest

from gpu_power_analysis import paths
from gpu_power_analysis import preprocess as pre

MISSING_IN_RAW = {"", "-", "Unknown Release Date"}

# Raw column -> parsed column that is missing exactly when the raw cell is.
ONE_TO_ONE = {
    "Name": "name",
    "Manufacturer": "manufacturer",
    "Architecture": "architecture",
    "Release_Date": "release_year",
    "Dedicated": "is_dedicated",
    "Notebook_GPU": "is_notebook",
    "Max_Power": "max_power_w",
    "TMUs": "tmus",
    "Core_Speed": "core_speed_mhz",
    "Boost_Clock": "boost_clock_mhz",
    "Memory": "memory_mb",
    "Memory_Bus": "memory_bus_bit",
    "Memory_Speed": "memory_speed_mhz",
    "Memory_Bandwidth": "memory_bandwidth_gb_s",
    "Memory_Type": "memory_type",
    "Process": "process_nm",
    "Texture_Rate": "texture_rate_gtexel_s",
}


@pytest.fixture(scope="module")
def stages() -> dict[str, pd.DataFrame]:
    text = pd.read_csv(paths.RAW_GPUS, dtype="str", keep_default_na=False)
    selected = pre.select_columns(pre.read_raw(paths.RAW_GPUS))
    parsed = pre.parse_columns(pre.clean_text(selected))
    rows, counts = pre.filter_rows(parsed)
    return {
        "text": text.apply(lambda column: column.str.strip()),
        "parsed": parsed,
        "counts": counts,
        "analysis": pre.derive_variables(rows),
    }


def test_selected_columns_exist_in_raw_file():
    header = pd.read_csv(paths.RAW_GPUS, nrows=0).columns
    assert set(pre.USED_COLUMNS) <= set(header)
    assert len(pre.USED_COLUMNS) == len(set(pre.USED_COLUMNS))


def test_read_raw_leaves_values_untouched():
    raw = pre.read_raw(paths.RAW_GPUS)
    assert raw["Release_Date"].str.startswith("\n").any()
    assert raw["Core_Speed"].str.strip().eq("-").any()


def test_parsing_keeps_every_row_in_order(stages):
    parsed, text = stages["parsed"], stages["text"]
    assert parsed["gpu_id"].tolist() == list(range(len(text)))


@pytest.mark.parametrize(("raw_column", "column"), sorted(ONE_TO_ONE.items()))
def test_no_value_lost_or_invented(stages, raw_column, column):
    raw_missing = stages["text"][raw_column].isin(MISSING_IN_RAW)
    assert (stages["parsed"][column].isna() == raw_missing).all()


def test_tmus_repair_satisfies_texture_identity(stages):
    assert pre.check_consistency(stages["parsed"])["texture_rate_mismatch"] == 0


def test_row_counts_never_increase(stages):
    counts, analysis = stages["counts"], stages["analysis"]
    assert counts["rows"].is_monotonic_decreasing
    assert counts["rows"].iloc[-1] == len(analysis)


def test_kept_rows_satisfy_every_filter(stages):
    analysis = stages["analysis"]
    kept = stages["parsed"].set_index("gpu_id").loc[analysis["gpu_id"]]
    assert kept["is_dedicated"].all()
    assert pre.match_multi_gpu_rules(kept["name"]).eq("").all()
    assert not kept["name"].str.contains("not released").any()


def test_values_are_carried_over_unchanged(stages):
    analysis = stages["analysis"]
    source = stages["parsed"].set_index("gpu_id").loc[analysis["gpu_id"]].reset_index()
    for column in ["max_power_w", "tmus", "memory_bandwidth_gb_s", "release_year"]:
        pd.testing.assert_series_equal(
            analysis[column], source[column], check_names=False
        )


def test_every_final_column_has_a_role_and_a_use(stages):
    assert list(stages["analysis"].columns) == list(pre.COLUMNS)
    for role, used_in, description in pre.COLUMNS.values():
        assert role and used_in and description


def test_needs_aux_power_threshold_is_strict(stages):
    rows = stages["parsed"].head(3).copy()
    rows["max_power_w"] = pd.array([75.0, 76.0, None], dtype="Float64")
    derived = pre.derive_variables(rows.reset_index(drop=True))
    assert derived["needs_aux_power"].tolist() == [False, True, pd.NA]


def test_pool_rare_keeps_missing():
    values = pd.Series(["GDDR5"] * 3 + ["HBM-2", None], dtype="category")
    result = pre.pool_rare(values, min_size=2, other="Other")
    assert result.tolist()[:4] == ["GDDR5", "GDDR5", "GDDR5", "Other"]
    assert pd.isna(result.iloc[4])


def test_pool_rare_counts_only_selected_rows():
    values = pd.Series(["A", "A", "A", "B", "B", "B"], dtype="category")
    counted = pd.Series([True, True, True, True, False, False])
    result = pre.pool_rare(values, min_size=2, other="Other", counted=counted)
    assert result.tolist() == ["A", "A", "A", "Other", "Other", "Other"]
