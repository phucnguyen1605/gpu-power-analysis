import pandas as pd
import pytest

from gpu_power_analysis import clean, parsers


def text(values: list[str | None]) -> pd.Series:
    return pd.Series(values, dtype="str", name="column")


@pytest.mark.parametrize(
    ("values", "unit", "prefix", "expected"),
    [
        (["738 MHz", None], "MHz", "", [738.0, pd.NA]),
        (["55nm"], "nm", "", [55.0]),
        (["141 Watts"], "Watts", "", [141.0]),
        (["$299"], "", "$", [299.0]),
        (["12 GPixel/s"], "GPixel/s", "", [12.0]),
    ],
)
def test_parse_number(values, unit, prefix, expected):
    result = parsers.parse_number(text(values), unit, prefix)
    assert result.tolist() == expected


def test_parse_number_rejects_wrong_unit():
    with pytest.raises(ValueError, match="not"):
        parsers.parse_number(text(["1.2 GHz"]), "MHz")


def test_parse_bandwidth_converts_mb():
    assert parsers.parse_bandwidth(text(["64GB/sec", "800MB/sec"])).tolist() == [
        64.0,
        0.8,
    ]


def test_parse_multiplied():
    assert parsers.parse_multiplied(text(["32", "16 (x2)"])).tolist() == [32.0, 32.0]
    assert parsers.parse_multiplied(text(["256KB(x2)"]), "KB").tolist() == [512.0]


def test_parse_resolution_and_psu():
    resolution = parsers.parse_resolution(text(["2560x1600", "1366 x 768"]))
    assert resolution["width"].tolist() == [2560.0, 1366.0]
    psu = parsers.parse_psu(text(["450 Watt & 38 Amps", "300 Watt"]))
    assert psu["watts"].tolist() == [450.0, 300.0]
    assert psu["amps"].tolist() == [38.0, pd.NA]


def test_parse_yes_no_rejects_other_values():
    with pytest.raises(ValueError):
        parsers.parse_yes_no(text(["Yes", "Maybe"]))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("GeForce GTX 680 Quad SLI", "sli_crossfire"),
        ("Radeon R9 295X2", "x2_board,dual_chip_model"),
        ("Radeon HD 8670D + 6670 Dual", "hybrid_graphics"),
        ("GeForce GTX Titan Z", "dual_chip_model"),
        ("GeForce GTX Titan Zotac Edition", ""),
        ("Quadro K2200", ""),
        ("Radeon R9 280X Sapphire Dual-X 3GB OC Edition", ""),
        ("Radeon R7 265 PowerColor TurboDuo OC 2GB Edition", ""),
        ("GeForce 9800 GX2", "x2_board"),
        ("Radeon HD 6870 PowerColor X2 Edition", "x2_board"),
        ("Radeon RX 580 HIS IceQ X2 OC 8GB", ""),
        ("GeForce GTX 1050 Inno3D Twin X2 2GB", ""),
        ("Tesla K10", "dual_chip_model"),
        ("FirePro S10000", "dual_chip_model"),
        ("Quadro NVS 420", "dual_chip_model"),
        ("NVS 4200M", ""),
        ("Radeon HD 6990", "dual_chip_model"),
        ("Radeon HD 6990M", ""),
        ("Radeon HD 7970 Asus ARES II Edition", "dual_chip_model"),
        ("GeForce GTX 660 Ti Colorful iGame ARES X 2GB Edition", ""),
    ],
)
def test_multi_gpu_rules(name, expected):
    assert clean.match_multi_gpu_rules(text([name])).iloc[0] == expected


def test_parse_multiplier_defaults_to_one():
    result = parsers.parse_multiplier(text(["16 (x2)", "32", None]))
    assert result.tolist() == [2.0, 1.0, 1.0]


@pytest.mark.parametrize(
    ("values", "prefix", "expected"),
    [
        (["DX 12", "DX 12.0", "DX 9.0c"], "DX ", ["12.0", "12.0", "9.0c"]),
        (["5", "5.0", "4.1"], "", ["5.0", "5.0", "4.1"]),
    ],
)
def test_normalize_version_merges_spellings(values, prefix, expected):
    assert parsers.normalize_version(text(values), prefix).tolist() == expected


def test_parse_number_rejects_other_currency_and_separators():
    for value in ["7.500.000 VND", "₫7500000", "$1,099", "299 USD"]:
        with pytest.raises(ValueError):
            parsers.parse_number(text([value]), prefix="$")
