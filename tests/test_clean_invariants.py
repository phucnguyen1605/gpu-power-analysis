"""Properties the cleaned table must always have, checked on the real raw file."""

import pandas as pd
import pytest

from gpu_power_analysis import clean, paths

MISSING_IN_RAW = {"", "-", "Unknown Release Date"}

# Columns whose missingness legitimately differs from their raw source:
# spec_multiplier defaults to 1, and a PSU listed only in watts has no amps.
NOT_ONE_TO_ONE = {"release_year", "spec_multiplier", "psu_recommended_amps"}

# Raw column -> a clean column that is missing exactly when the raw cell is.
ONE_TO_ONE = {
    raw: column
    for column, (raw, _) in clean.COLUMNS.items()
    if raw and raw != "Name" and column not in NOT_ONE_TO_ONE
}


@pytest.fixture(scope="module")
def tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = clean.read_raw(paths.RAW_GPUS)
    text = pd.read_csv(paths.RAW_GPUS, dtype="str", keep_default_na=False)
    parsed = clean.parse_columns(raw)
    return text.apply(lambda column: column.str.strip()), parsed


def test_rows_kept_in_order(tables):
    text, parsed = tables
    assert len(parsed) == len(text)
    assert parsed["gpu_id"].tolist() == list(range(len(text)))


def test_every_raw_column_is_represented():
    sources = {raw for raw, _ in clean.COLUMNS.values() if raw}
    header = pd.read_csv(paths.RAW_GPUS, nrows=0).columns
    assert set(header) <= sources


@pytest.mark.parametrize(("raw_column", "column"), sorted(ONE_TO_ONE.items()))
def test_no_value_lost_or_invented(tables, raw_column, column):
    text, parsed = tables
    raw_missing = text[raw_column].isin(MISSING_IN_RAW)
    assert (parsed[column].isna() == raw_missing).all()
