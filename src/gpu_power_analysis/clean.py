"""Step-by-step cleaning of the raw GPU file into a typed, unfiltered table.

The result keeps every raw row and every raw column. Rows that a study may
want to exclude are flagged, never removed, so any research question on this
dataset can start from it.

Steps:
    1. read_raw                 read text, trim whitespace, mark missing values
    2. find_multi_gpu_candidates  propose multi-GPU products from their names
    3. apply_review             take the final decision from the manual review
    4. parse_columns            convert every column to its proper type
    5. add_flags                flag multi-GPU, unreleased and duplicate rows
    6. describe_columns         build the data dictionary
"""

import logging
from pathlib import Path

import pandas as pd

from gpu_power_analysis import parsers, paths

logger = logging.getLogger(__name__)

# Cell contents that mean "not recorded", in any column.
MISSING_TOKENS = ["", "-"]
UNKNOWN_DATE = "Unknown Release Date"

# Text that was mis-encoded in the raw file (UTF-8 read as Latin-1).
ENCODING_FIXES = {"Â²": "²"}

# Name patterns of multi-GPU products. Each rule is narrow on purpose: words
# such as "Quad", "Dual" or "Duo" also appear in single-GPU names ("Quadro",
# "Dual-X" coolers, "TurboDuo"), and every candidate is reviewed by hand.
MULTI_GPU_RULES = {
    "sli_crossfire": r"(?i)\bSLI\b|\bcross ?fire\b|\b[2-4]-way\b",
    "hybrid_graphics": r"\s\+\s",
    "x2_board": r"(?i)\d\s?X2\b",
    "quadro_plex": r"\bPlex\b",
    "dual_chip_model": (
        r"(?i)GTX 690|GTX 590|GTX 295|Titan Z\b|HD 5970|HD 6990|HD 7990|HD 8990"
        r"|Pro Duo|Devil 13|\bARES\b|\bMARS\b|R9 295"
    ),
}

REVIEW_COLUMNS = ["gpu_id", "name", "matched_rules", "exclude", "reviewed", "note"]

# Clean column -> (raw column it comes from, meaning). Also fixes column order.
COLUMNS = {
    "gpu_id": ("", "Row position in the raw file (0-based), stable identifier"),
    "name": ("Name", "Product name"),
    "manufacturer": ("Manufacturer", "GPU vendor"),
    "architecture": ("Architecture", "GPU chip / architecture"),
    "release_date": ("Release_Date", "Release date"),
    "release_year": ("Release_Date", "Release year"),
    "is_dedicated": ("Dedicated", "Dedicated (discrete) GPU"),
    "is_integrated": ("Integrated", "Integrated GPU"),
    "is_notebook": ("Notebook_GPU", "Laptop GPU"),
    "supports_sli_crossfire": ("SLI_Crossfire", "Can be paired with other cards"),
    "max_power_w": ("Max_Power", "Rated maximum power (W)"),
    "core_speed_mhz": ("Core_Speed", "Base core clock (MHz)"),
    "boost_clock_mhz": ("Boost_Clock", "Boost clock (MHz)"),
    "memory_mb": ("Memory", "Video memory (MB)"),
    "memory_bus_bit": (
        "Memory_Bus",
        "Memory bus width (bit), corrected by spec_multiplier",
    ),
    "memory_speed_mhz": ("Memory_Speed", "Memory clock (MHz)"),
    "memory_bandwidth_gb_s": ("Memory_Bandwidth", "Memory bandwidth (GB/s)"),
    "memory_type": ("Memory_Type", "Memory technology"),
    "tmus": ("TMUs", "Texture mapping units, corrected by spec_multiplier"),
    "rops": ("ROPs", "Render output units"),
    "spec_multiplier": (
        "ROPs",
        "N from 'a (xN)' in ROPs; TMUs and Memory_Bus were listed divided by N",
    ),
    "process_nm": ("Process", "Manufacturing process (nm)"),
    "pixel_rate_gpixel_s": ("Pixel_Rate", "Pixel fill rate (GPixel/s)"),
    "texture_rate_gtexel_s": ("Texture_Rate", "Texture fill rate (GTexel/s)"),
    "l2_cache_kb": ("L2_Cache", "L2 cache (KB)"),
    "release_price_usd": ("Release_Price", "Launch price (USD)"),
    "dvi_ports": ("DVI_Connection", "DVI outputs"),
    "hdmi_ports": ("HDMI_Connection", "HDMI outputs"),
    "vga_ports": ("VGA_Connection", "VGA outputs"),
    "displayport_ports": ("DisplayPort_Connection", "DisplayPort outputs"),
    "best_resolution_width": ("Best_Resolution", "Recommended resolution, width"),
    "best_resolution_height": ("Best_Resolution", "Recommended resolution, height"),
    "max_resolution_width": ("Resolution_WxH", "Maximum resolution, width"),
    "max_resolution_height": ("Resolution_WxH", "Maximum resolution, height"),
    "psu_recommended_w": ("PSU", "Recommended system power supply (W)"),
    "psu_recommended_amps": ("PSU", "Recommended +12 V current (A)"),
    "power_connector": ("Power_Connector", "Auxiliary power connector"),
    "direct_x": ("Direct_X", "DirectX version, e.g. '12.0', '9.0c'"),
    "open_gl": ("Open_GL", "OpenGL version"),
    "shader_model": ("Shader", "Shader Model version, e.g. '5.0' (not a core count)"),
    "multi_gpu_rules": ("Name", "Multi-GPU name rules matched, if any"),
    "is_multi_gpu": ("Name", "Multi-GPU product, per the manual review"),
    "is_unreleased": ("Name", "Marked '[not released]'"),
    "is_exact_duplicate": ("", "Identical to an earlier row on every raw column"),
}


def read_raw(path: Path) -> pd.DataFrame:
    """Step 1: read every column as text, trim it and mark missing values.

    Reading as text means no value is converted before its format is checked
    in step 4.
    """
    raw = pd.read_csv(path, dtype="str", keep_default_na=False)
    raw = raw.apply(lambda column: column.str.strip())
    raw = raw.replace(MISSING_TOKENS, pd.NA)
    raw["Release_Date"] = raw["Release_Date"].replace(UNKNOWN_DATE, pd.NA)
    for wrong, right in ENCODING_FIXES.items():
        raw["Name"] = raw["Name"].str.replace(wrong, right, regex=False)
    return raw.rename_axis("gpu_id").reset_index()


def match_multi_gpu_rules(names: pd.Series) -> pd.Series:
    """Comma-separated names of the multi-GPU rules each name matches."""
    matches = pd.DataFrame(
        {rule: names.str.contains(pattern) for rule, pattern in MULTI_GPU_RULES.items()}
    ).fillna(False)
    return matches.apply(lambda row: ",".join(row.index[row]), axis=1)


def find_multi_gpu_candidates(raw: pd.DataFrame) -> pd.DataFrame:
    """Step 2: list rows whose name matches a multi-GPU rule."""
    rules = match_multi_gpu_rules(raw["Name"])
    candidates = raw.loc[rules != "", ["gpu_id", "Name"]].rename(
        columns={"Name": "name"}
    )
    return candidates.assign(
        matched_rules=rules[rules != ""], exclude=True, reviewed=False, note=""
    )[REVIEW_COLUMNS]


def apply_review(candidates: pd.DataFrame, review_path: Path) -> set[int]:
    """Step 3: decide which rows are multi-GPU, using the manual review file.

    The review file is the authority. A reviewer sets ``exclude`` to False for
    a false positive, or adds a row for a product no rule found. Candidates
    not yet in the file count as multi-GPU. On the first run the file is
    created from the candidates.

    Returns:
        ``gpu_id`` values of multi-GPU products.
    """
    if not review_path.exists():
        candidates.to_csv(review_path, index=False)
        logger.warning("Created %s; review it by hand", review_path)
    review = pd.read_csv(
        review_path,
        dtype={"gpu_id": "int64", "exclude": "boolean", "reviewed": "boolean"},
    )
    not_in_file = candidates.loc[~candidates["gpu_id"].isin(review["gpu_id"])]
    not_reviewed = review.loc[~review["reviewed"].fillna(False)]
    if len(not_in_file):
        logger.warning("%d candidates missing from the review file", len(not_in_file))
    if len(not_reviewed):
        logger.warning("%d rows of the review file not reviewed yet", len(not_reviewed))
    excluded = review.loc[review["exclude"].fillna(True), "gpu_id"]
    return set(excluded) | set(not_in_file["gpu_id"])


def parse_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Step 4: convert every raw column into its proper type."""
    num = parsers.parse_number
    best_resolution = parsers.parse_resolution(raw["Best_Resolution"])
    max_resolution = parsers.parse_resolution(raw["Resolution_WxH"])
    psu = parsers.parse_psu(raw["PSU"])
    release_date = parsers.parse_date(raw["Release_Date"])
    # Where ROPs read "a (xN)", the source listed TMUs and Memory_Bus divided by
    # N; Texture_Rate and Memory_Bandwidth only agree with them once multiplied.
    multiplier = parsers.parse_multiplier(raw["ROPs"])

    return pd.DataFrame(
        {
            "gpu_id": raw["gpu_id"].astype("int64"),
            "name": raw["Name"],
            "manufacturer": raw["Manufacturer"].astype("category"),
            "architecture": raw["Architecture"],
            "release_date": release_date,
            "release_year": release_date.dt.year.astype("Int64"),
            "is_dedicated": parsers.parse_yes_no(raw["Dedicated"]),
            "is_integrated": parsers.parse_yes_no(raw["Integrated"]),
            "is_notebook": parsers.parse_yes_no(raw["Notebook_GPU"]),
            "supports_sli_crossfire": parsers.parse_yes_no(raw["SLI_Crossfire"]),
            "max_power_w": num(raw["Max_Power"], "Watts"),
            "core_speed_mhz": num(raw["Core_Speed"], "MHz"),
            "boost_clock_mhz": num(raw["Boost_Clock"], "MHz"),
            "memory_mb": num(raw["Memory"], "MB"),
            "memory_bus_bit": num(raw["Memory_Bus"], "Bit") * multiplier,
            "memory_speed_mhz": num(raw["Memory_Speed"], "MHz"),
            "memory_bandwidth_gb_s": parsers.parse_bandwidth(raw["Memory_Bandwidth"]),
            "memory_type": raw["Memory_Type"].astype("category"),
            "tmus": num(raw["TMUs"]) * multiplier,
            "rops": parsers.parse_multiplied(raw["ROPs"]),
            "spec_multiplier": multiplier,
            "process_nm": num(raw["Process"], "nm"),
            "pixel_rate_gpixel_s": num(raw["Pixel_Rate"], "GPixel/s"),
            "texture_rate_gtexel_s": num(raw["Texture_Rate"], "GTexel/s"),
            "l2_cache_kb": parsers.parse_multiplied(raw["L2_Cache"], "KB"),
            "release_price_usd": num(raw["Release_Price"], prefix="$"),
            "dvi_ports": num(raw["DVI_Connection"]),
            "hdmi_ports": num(raw["HDMI_Connection"]),
            "vga_ports": num(raw["VGA_Connection"]),
            "displayport_ports": num(raw["DisplayPort_Connection"]),
            "best_resolution_width": best_resolution["width"],
            "best_resolution_height": best_resolution["height"],
            "max_resolution_width": max_resolution["width"],
            "max_resolution_height": max_resolution["height"],
            "psu_recommended_w": psu["watts"],
            "psu_recommended_amps": psu["amps"],
            "power_connector": raw["Power_Connector"].astype("category"),
            "direct_x": parsers.normalize_version(raw["Direct_X"], prefix="DX "),
            "open_gl": parsers.normalize_version(raw["Open_GL"]),
            "shader_model": parsers.normalize_version(raw["Shader"]),
        }
    )


def check_consistency(clean: pd.DataFrame, tolerance: float = 0.1) -> dict[str, int]:
    """Count rows whose related columns disagree by more than ``tolerance``.

    Two identities hold for real GPUs:
      * texture rate = TMUs x clock (boost clock when listed, else base clock);
      * bandwidth = bus width / 8 x memory clock x k, where the data-rate
        factor k is fixed for each memory type (taken as the type's median).
    A row that breaks one of them carries a data error in one of its columns.
    Texture rates are rounded to whole GTexel/s in the raw file, so a gap of
    up to 0.5 is accepted whatever its relative size.
    """
    clock = clean["boost_clock_mhz"].fillna(clean["core_speed_mhz"])
    expected_texture = clean["tmus"] * clock / 1000
    texture_gap = (clean["texture_rate_gtexel_s"] - expected_texture).abs()
    data_rate = (
        clean["memory_bandwidth_gb_s"]
        * 1000
        / (clean["memory_bus_bit"] / 8 * clean["memory_speed_mhz"])
    )
    typical = data_rate.groupby(clean["memory_type"], observed=True).transform("median")
    return {
        "texture_rate_mismatch": int(
            ((texture_gap > 0.5) & (texture_gap > tolerance * expected_texture)).sum()
        ),
        "bandwidth_mismatch": int(((data_rate / typical - 1).abs() > tolerance).sum()),
    }


def add_flags(
    clean: pd.DataFrame, raw: pd.DataFrame, multi_gpu_ids: set[int]
) -> pd.DataFrame:
    """Step 5: flag rows a study may want to exclude; no row is removed."""
    raw_columns = [column for column in raw.columns if column != "gpu_id"]
    flagged = clean.assign(
        multi_gpu_rules=match_multi_gpu_rules(raw["Name"]),
        is_multi_gpu=clean["gpu_id"].isin(multi_gpu_ids).astype("boolean"),
        is_unreleased=raw["Name"]
        .str.contains(r"(?i)\[not released\]")
        .astype("boolean"),
        is_exact_duplicate=raw.duplicated(subset=raw_columns).astype("boolean"),
    )
    return flagged[list(COLUMNS)]


def describe_columns(clean: pd.DataFrame) -> pd.DataFrame:
    """Step 6: data dictionary with source, type and missing share per column."""
    return pd.DataFrame(
        {
            "column": list(COLUMNS),
            "source": [source for source, _ in COLUMNS.values()],
            "dtype": [str(clean[column].dtype) for column in COLUMNS],
            "missing": [int(clean[column].isna().sum()) for column in COLUMNS],
            "missing_pct": [
                round(100 * clean[column].isna().mean(), 1) for column in COLUMNS
            ],
            "description": [description for _, description in COLUMNS.values()],
        }
    )


def load_all_clean(path: Path = paths.CLEAN_ALL_PARQUET) -> pd.DataFrame:
    """Read the cleaned table back with its dtypes (Parquet keeps them)."""
    return pd.read_parquet(path)
