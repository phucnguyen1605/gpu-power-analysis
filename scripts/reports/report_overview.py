"""Compute every number and table row quoted in the overview section of the report.

The numbers and variable rows are written as LaTeX files under report/generated/,
so the overview section stays synchronized with the raw CSV file.

Usage:
    python scripts/reports/report_overview.py

Outputs (report/generated/):
    overview_numbers.tex            numbers quoted in the text, as macros
    overview_variables_rows.tex     body of the 34-variable overview table
"""

import logging

import pandas as pd

from gpu_power_analysis import parsers, paths
from gpu_power_analysis import preprocess as pre

logger = logging.getLogger("report_overview")

GENERATED = paths.ROOT / "report" / "generated"
NUMBERS_OUTPUT = GENERATED / "overview_numbers.tex"
VARIABLES_OUTPUT = GENERATED / "overview_variables_rows.tex"

# (column_name, stat_type, unit, description, has_hidden_missing)
VariableSpec = tuple[str, str, str, str, bool]

VARIABLE_GROUPS: list[tuple[str, list[VariableSpec]]] = [
    (
        "Nhóm 1: Định danh và phân loại chung",
        [
            ("Name", "Định danh", "-", "Tên thương mại của GPU", False),
            (
                "Manufacturer",
                "Định tính",
                "-",
                "Hãng sản xuất (Nvidia, AMD, Intel, ATI)",
                False,
            ),
            ("Release_Date", "Thời gian", "-", "Ngày phát hành", True),
            (
                "Release_Price",
                "Định lượng liên tục",
                "USD",
                "Giá niêm yết khi phát hành",
                False,
            ),
            (
                "Dedicated",
                "Định tính nhị phân",
                "-",
                "Card đồ họa rời (Yes/No)",
                False,
            ),
            (
                "Integrated",
                "Định tính nhị phân",
                "-",
                "Card đồ họa tích hợp (Yes/No)",
                False,
            ),
            (
                "Notebook_GPU",
                "Định tính nhị phân",
                "-",
                "GPU dành cho laptop (Yes/No)",
                False,
            ),
            (
                "SLI_Crossfire",
                "Định tính nhị phân",
                "-",
                "Hỗ trợ ghép đa GPU SLI/CrossFire (Yes/No)",
                False,
            ),
        ],
    ),
    (
        "Nhóm 2: Kiến trúc và nhân xử lý",
        [
            ("Architecture", "Định tính", "-", "Tên kiến trúc GPU", False),
            (
                "Process",
                "Định lượng rời rạc",
                "nm",
                "Tiến trình công nghệ bán dẫn",
                False,
            ),
            (
                "Core_Speed",
                "Định lượng liên tục",
                "MHz",
                "Xung nhịp cơ bản của nhân",
                True,
            ),
            (
                "Boost_Clock",
                "Định lượng liên tục",
                "MHz",
                "Xung nhịp tăng cường tối đa",
                False,
            ),
            (
                "L2_Cache",
                "Định lượng rời rạc",
                "KB",
                "Bộ nhớ đệm L2 (nhiều card ghi 0KB; một số ghi dạng (x$N$))",
                False,
            ),
        ],
    ),
    (
        "Nhóm 3: Bộ nhớ đồ họa (VRAM)",
        [
            ("Memory", "Định lượng rời rạc", "MB", "Dung lượng bộ nhớ đồ họa", False),
            (
                "Memory_Type",
                "Định tính",
                "-",
                "Chuẩn bộ nhớ (GDDR3, GDDR5, DDR3, \\dots)",
                False,
            ),
            ("Memory_Speed", "Định lượng liên tục", "MHz", "Xung nhịp bộ nhớ", False),
            ("Memory_Bus", "Định lượng rời rạc", "Bit", "Độ rộng bus bộ nhớ", False),
            (
                "Memory_Bandwidth",
                "Định lượng liên tục",
                "GB/sec, MB/sec",
                "Băng thông bộ nhớ tối đa",
                False,
            ),
        ],
    ),
    (
        "Nhóm 4: Đơn vị xử lý đồ họa và chuẩn API",
        [
            (
                "ROPs",
                "Định lượng rời rạc",
                "-",
                "Số đơn vị xuất điểm ảnh (một số ghi dạng $a$ (x$N$))",
                False,
            ),
            (
                "TMUs",
                "Định lượng rời rạc",
                "-",
                "Số đơn vị ánh xạ kết cấu",
                False,
            ),
            (
                "Pixel_Rate",
                "Định lượng liên tục",
                "GPixel/s",
                "Tốc độ điền điểm ảnh",
                False,
            ),
            (
                "Texture_Rate",
                "Định lượng liên tục",
                "GTexel/s",
                "Tốc độ xử lý kết cấu",
                False,
            ),
            ("Shader", "Định tính thứ bậc", "-", "Phiên bản Shader Model", False),
            (
                "Direct_X",
                "Định tính thứ bậc",
                "-",
                "Phiên bản DirectX hỗ trợ",
                False,
            ),
            ("Open_GL", "Định tính thứ bậc", "-", "Phiên bản OpenGL hỗ trợ", False),
        ],
    ),
    (
        "Nhóm 5: Điện năng",
        [
            (
                "Max_Power",
                "Định lượng liên tục",
                "Watts",
                "\\textbf{Công suất tiêu thụ tối đa (biến mục tiêu)}",
                False,
            ),
            (
                "PSU",
                "Chuỗi hỗn hợp",
                "Watt, Amps",
                "Bộ nguồn khuyến nghị (dạng 450 Watt \\& 38 Amps)",
                False,
            ),
            (
                "Power_Connector",
                "Định tính",
                "-",
                "Đầu cấp nguồn phụ (6-pin, 8-pin, \\dots; None là không có)",
                False,
            ),
        ],
    ),
    (
        "Nhóm 6: Độ phân giải và cổng xuất hình",
        [
            (
                "Best_Resolution",
                "Định tính thứ bậc",
                "WxH (px)",
                "Độ phân giải tối ưu",
                False,
            ),
            (
                "Resolution_WxH",
                "Định tính thứ bậc",
                "WxH (px)",
                "Độ phân giải tối đa hỗ trợ",
                False,
            ),
            ("DVI_Connection", "Định lượng rời rạc", "cổng", "Số cổng DVI", False),
            ("HDMI_Connection", "Định lượng rời rạc", "cổng", "Số cổng HDMI", False),
            ("VGA_Connection", "Định lượng rời rạc", "cổng", "Số cổng VGA", False),
            (
                "DisplayPort_Connection",
                "Định lượng rời rạc",
                "cổng",
                "Số cổng DisplayPort",
                False,
            ),
        ],
    ),
]


def percent(part: float, whole: float) -> str:
    """Format a percentage with two decimals and a Vietnamese decimal comma."""
    return f"{100 * part / whole:.2f}".replace(".", ",")


def clean_raw_text() -> pd.DataFrame:
    """Read the raw CSV as strings, strip whitespace, and mark missing tokens."""
    raw = pre.read_raw(paths.RAW_GPUS).drop(columns="gpu_id")
    cleaned = raw.apply(lambda column: column.str.strip())
    cleaned = cleaned.replace(pre.MISSING_TOKENS, pd.NA)
    cleaned["Release_Date"] = cleaned["Release_Date"].replace(pre.UNKNOWN_DATE, pd.NA)
    return cleaned


def tex_var_name(name: str) -> str:
    """Escape underscores with allowbreak so long names wrap inside narrow columns."""
    return name.replace("_", r"\_\allowbreak{}")


def numbers() -> dict[str, str | int]:
    """Compute every number quoted in the overview section, keyed by macro name."""
    raw = pre.read_raw(paths.RAW_GPUS).drop(columns="gpu_id")
    stripped = raw.apply(lambda column: column.str.strip())
    cleaned = clean_raw_text()
    default_read = pd.read_csv(paths.RAW_GPUS)

    n_rows, n_cols = raw.shape
    numeric_cols = sum(
        pd.api.types.is_numeric_dtype(default_read[column])
        for column in default_read.columns
    )
    dates = parsers.parse_date(cleaned["Release_Date"]).dropna()
    core_dash = int(stripped["Core_Speed"].eq("-").sum())
    core_blank = int(stripped["Core_Speed"].eq("").sum())
    core_missing = int(cleaned["Core_Speed"].isna().sum())
    date_unknown = int(stripped["Release_Date"].eq(pre.UNKNOWN_DATE).sum())

    makers = cleaned["Manufacturer"].value_counts()
    dedicated = parsers.parse_yes_no(cleaned["Dedicated"])
    notebook = parsers.parse_yes_no(cleaned["Notebook_GPU"])
    power_observed = int(cleaned["Max_Power"].notna().sum())
    power_missing = int(cleaned["Max_Power"].isna().sum())

    return {
        "RowsRaw": n_rows,
        "ColsRaw": n_cols,
        "DefaultNumericCols": int(numeric_cols),
        "DefaultObjectCols": int(n_cols - numeric_cols),
        "DateValid": int(len(dates)),
        "DateMin": dates.min().strftime("%d/%m/%Y"),
        "DateMax": dates.max().strftime("%d/%m/%Y"),
        "DateUnknown": date_unknown,
        "DateUnknownPct": percent(date_unknown, n_rows),
        "CoreSpeedDash": core_dash,
        "CoreSpeedBlank": core_blank,
        "CoreSpeedMissing": core_missing,
        "CoreSpeedMissingPct": percent(core_missing, n_rows),
        "ManufacturerLevels": int(cleaned["Manufacturer"].nunique()),
        "NvidiaCount": int(makers["Nvidia"]),
        "NvidiaPct": percent(makers["Nvidia"], n_rows),
        "AmdCount": int(makers["AMD"]),
        "AmdPct": percent(makers["AMD"], n_rows),
        "IntelCount": int(makers["Intel"]),
        "IntelPct": percent(makers["Intel"], n_rows),
        "AtiCount": int(makers["ATI"]),
        "AtiPct": percent(makers["ATI"], n_rows),
        "DedicatedYes": int(dedicated.eq(True).sum()),
        "DedicatedNo": int(dedicated.eq(False).sum()),
        "DedicatedMissing": int(dedicated.isna().sum()),
        "NotebookYes": int(notebook.eq(True).sum()),
        "NotebookNo": int(notebook.eq(False).sum()),
        "PowerObserved": power_observed,
        "PowerMissing": power_missing,
        "PowerMissingPct": percent(power_missing, n_rows),
        "PriceMissingPct": percent(cleaned["Release_Price"].isna().sum(), n_rows),
        "DpMissingPct": percent(
            cleaned["DisplayPort_Connection"].isna().sum(), n_rows
        ),
        "BoostMissingPct": percent(cleaned["Boost_Clock"].isna().sum(), n_rows),
    }


def variable_rows(cleaned: pd.DataFrame, default_read: pd.DataFrame) -> str:
    """Build the LaTeX body rows for the 34-variable overview table."""
    listed_names = [spec[0] for _, specs in VARIABLE_GROUPS for spec in specs]
    if len(listed_names) != len(set(listed_names)) or set(listed_names) != set(
        cleaned.columns
    ):
        msg = "VARIABLE_GROUPS must match all raw columns without duplicates."
        raise ValueError(msg)

    n_rows = len(cleaned)
    lines: list[str] = []
    for group_idx, (group_title, specs) in enumerate(VARIABLE_GROUPS):
        if group_idx > 0:
            lines.append(r"\addlinespace[6pt]")
        lines.append(
            rf"\multicolumn{{6}}{{@{{}}l@{{}}}}{{\textbf{{{group_title}}}}} \\*"
        )
        for name, stat_type, unit, description, has_hidden_missing in specs:
            raw_type = (
                "Số thực"
                if pd.api.types.is_numeric_dtype(default_read[name])
                else "Kí tự"
            )
            missing_pct = percent(int(cleaned[name].isna().sum()), n_rows)
            if has_hidden_missing:
                missing_pct = f"{missing_pct}$^{{*}}$"
            lines.append(
                f"{tex_var_name(name)} & {raw_type} & {stat_type} & "
                rf"{unit} & {description} & {missing_pct} \\"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write overview_numbers.tex and overview_variables_rows.tex."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    header = (
        "% Generated by scripts/reports/report_overview.py; do not edit by hand.\n"
    )
    macro_values = numbers()
    lines = [
        f"\\newcommand{{\\Ovw{name}}}{{{value}}}"
        for name, value in macro_values.items()
    ]
    GENERATED.mkdir(parents=True, exist_ok=True)
    NUMBERS_OUTPUT.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")

    cleaned = clean_raw_text()
    default_read = pd.read_csv(paths.RAW_GPUS)
    VARIABLES_OUTPUT.write_text(
        header + variable_rows(cleaned, default_read), encoding="utf-8"
    )
    logger.info(
        "Wrote %d numbers to %s and %d variables to %s",
        len(lines),
        NUMBERS_OUTPUT.relative_to(paths.ROOT),
        len(cleaned.columns),
        VARIABLES_OUTPUT.relative_to(paths.ROOT),
    )


if __name__ == "__main__":
    main()
