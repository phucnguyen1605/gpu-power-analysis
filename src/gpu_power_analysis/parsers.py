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


def parse_multiplied(values: pd.Series, unit: str = "") -> pd.Series:
    """Parse ``"32"`` -> 32 and ``"16 (x2)"`` -> 32.

    The ``(xN)`` suffix is a multiplier, not a GPU count: the single-GPU
    Radeon RX 580 lists its ROPs as ``"16 (x2)"`` and has 32 ROPs. L2 cache
    uses the same notation, e.g. ``"256KB(x2)"``.
    """
    pattern = rf"^(?P<base>{NUMBER})\s?{re.escape(unit)}(?:\s?\(x(?P<factor>\d+)\))?$"
    parts = values.str.extract(pattern)
    _check_all_parsed(values, parts["base"], f"'<number>{unit}' or '... (xN)'")
    factor = pd.to_numeric(parts["factor"]).fillna(1)
    return (pd.to_numeric(parts["base"]) * factor).astype("Float64")


def parse_multiplier(values: pd.Series) -> pd.Series:
    """Return N from an ``"a (xN)"`` suffix, or 1 when there is no suffix.

    Missing values give 1 as well, so the result can always be multiplied.
    """
    factor = values.str.extract(r"\(x(\d+)\)$", expand=False)
    return pd.to_numeric(factor).fillna(1).astype("Float64")


def normalize_version(values: pd.Series, prefix: str = "") -> pd.Series:
    """Write versions one way: ``"DX 12"`` and ``"DX 12.0"`` both become ``"12.0"``.

    A trailing letter is kept (``"DX 9.0c"`` -> ``"9.0c"``).
    """
    parts = values.str.extract(
        rf"^{re.escape(prefix)}(?P<major>\d+)(?:\.(?P<minor>\d+))?(?P<letter>[a-z]?)$"
    )
    _check_all_parsed(values, parts["major"], f"'{prefix}<major>[.<minor>][letter]'")
    version = (
        parts["major"] + "." + parts["minor"].fillna("0") + parts["letter"].fillna("")
    )
    return version.astype("category")


def parse_resolution(values: pd.Series) -> pd.DataFrame:
    """Split ``"2560x1600"`` or ``"1366 x 768"`` into width and height."""
    parts = values.str.extract(r"^(?P<width>\d+)\s?x\s?(?P<height>\d+)$")
    _check_all_parsed(values, parts["width"], "'<width>x<height>'")
    return parts.apply(pd.to_numeric).astype("Float64")


def parse_psu(values: pd.Series) -> pd.DataFrame:
    """Split ``"450 Watt & 38 Amps"`` into watts and amps; amps may be absent."""
    parts = values.str.extract(r"^(?P<watts>\d+) Watt(?: & (?P<amps>\d+) Amps)?$")
    _check_all_parsed(values, parts["watts"], "'<n> Watt[ & <n> Amps]'")
    return parts.apply(pd.to_numeric).astype("Float64")


def parse_yes_no(values: pd.Series) -> pd.Series:
    """Parse ``"Yes"``/``"No"`` into a nullable boolean."""
    booleans = values.map({"Yes": True, "No": False})
    _check_all_parsed(values, booleans, "'Yes' or 'No'")
    return booleans.astype("boolean")


def parse_date(values: pd.Series) -> pd.Series:
    """Parse ``"01-Mar-2009"``; raises on any other format."""
    return pd.to_datetime(values, format="%d-%b-%Y")
