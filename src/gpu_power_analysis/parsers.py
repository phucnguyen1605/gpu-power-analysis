"""Convert raw specification strings into typed values.

Every parser checks that each non-missing value matches the expected format
and raises ``ValueError`` otherwise. A value is never silently turned into a
missing one, so an unexpected format in the raw file stops the pipeline.
"""

import re

import pandas as pd

NUMBER = r"\d+(?:\.\d+)?"

# Vendors quote memory bandwidth in decimal units (1 GB/s = 1000 MB/s).
MB_PER_GB = 1000


def _check_all_parsed(values: pd.Series, parsed: pd.Series, expected: str) -> None:
    """Raise if a value was present in ``values`` but could not be parsed."""
    failed = values.notna() & parsed.isna()
    if failed.any():
        examples = values[failed].unique()[:5].tolist()
        raise ValueError(
            f"{values.name}: {int(failed.sum())} values are not {expected}, "
            f"e.g. {examples}"
        )


def parse_number(values: pd.Series, unit: str = "", prefix: str = "") -> pd.Series:
    """Parse ``"738 MHz"`` -> 738, ``"55nm"`` -> 55, ``"$299"`` -> 299.

    Args:
        values: Raw strings; missing values stay missing.
        unit: Text after the number (an optional space is allowed before it).
        prefix: Text before the number, such as ``"$"``.

    Returns:
        Nullable float series.
    """
    pattern = rf"^{re.escape(prefix)}({NUMBER})\s?{re.escape(unit)}$"
    numbers = values.str.extract(pattern, expand=False)
    _check_all_parsed(values, numbers, f"'{prefix}<number>{unit}'")
    return pd.to_numeric(numbers).astype("Float64")


def parse_bandwidth(values: pd.Series) -> pd.Series:
    """Parse ``"64GB/sec"`` -> 64 and ``"800MB/sec"`` -> 0.8 (GB/s)."""
    parts = values.str.extract(rf"^(?P<number>{NUMBER})(?P<unit>GB|MB)/sec$")
    _check_all_parsed(values, parts["number"], "'<number>GB/sec' or 'MB/sec'")
    scale = parts["unit"].map({"GB": 1.0, "MB": 1.0 / MB_PER_GB})
    return (pd.to_numeric(parts["number"]) * scale).astype("Float64")


def parse_multiplier(values: pd.Series) -> pd.Series:
    """Return N from an ``"a (xN)"`` value, or 1 for a plain count ``"a"``.

    Missing values give 1 as well, so the result can always be multiplied.
    Raises on any other format.
    """
    parts = values.str.extract(r"^(?P<count>\d+)(?: \(x(?P<factor>[1-9]\d*)\))?$")
    _check_all_parsed(values, parts["count"], "a count or 'a (xN)'")
    return pd.to_numeric(parts["factor"]).fillna(1).astype("Float64")


def parse_yes_no(values: pd.Series) -> pd.Series:
    """Parse ``"Yes"``/``"No"`` into a nullable boolean."""
    booleans = values.map({"Yes": True, "No": False})
    _check_all_parsed(values, booleans, "'Yes' or 'No'")
    return booleans.astype("boolean")


def parse_date(values: pd.Series) -> pd.Series:
    """Parse ``"01-Mar-2009"``; raises on any other format."""
    return pd.to_datetime(values, format="%d-%b-%Y")
