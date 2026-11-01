# GPU Power Analysis

> **Phân tích thống kê các yếu tố ảnh hưởng đến công suất thiết kế tối đa của GPU rời
> và kiểm chứng ngoài mẫu với GPU thế hệ 2018–2025**

Dữ liệu: `All_GPUs.csv` (Kaggle, *Computer Parts*), 3406 GPU phát hành 1998–2017.

## Cài đặt

Cần Python 3.13.

```bash
python3.13 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # đúng phiên bản thư viện nhóm đang dùng
pip install -e . --no-deps           # để import được package gpu_power_analysis
```

## Chạy

```bash
python scripts/01_clean_raw.py       # bước 1: làm sạch dữ liệu thô
python scripts/02_build_dataset.py   # bước 2: dữ liệu cho đề tài
pytest                               # chạy test
ruff format . && ruff check .        # format và kiểm tra code trước khi commit
```

## Cấu trúc

```
gpu-power-analysis/
├── data/
│   ├── raw/All_GPUs.csv                 dữ liệu gốc, không bao giờ sửa
│   ├── interim/                         bước 1: dữ liệu sạch dùng chung (do script sinh ra)
│   └── processed/                       bước 2: dữ liệu của đề tài (do script sinh ra)
├── src/gpu_power_analysis/
│   ├── paths.py                         đường dẫn tới mọi file
│   ├── parsers.py                       các hàm đổi chuỗi thành số, ngày, Yes/No
│   ├── clean.py                         bước 1: 4 bước làm sạch, mỗi bước một hàm
│   └── dataset.py                       bước 2: lọc dòng, chọn và thêm biến
├── scripts/01_clean_raw.py              chạy bước 1, in kết quả từng bước con
├── scripts/02_build_dataset.py          chạy bước 2, in kết quả từng bước con
├── tests/test_parsers.py                test cho các hàm đọc giá trị
├── tests/test_clean_invariants.py       test file sạch không mất hay tự sinh dữ liệu
└── tests/test_dataset.py                test bước 2
```

## Bước làm sạch (`scripts/01_clean_raw.py`)

Kết quả **giữ đủ 3406 dòng và 34 biến gốc**. Bước này chỉ làm những việc khách quan;
các dòng có thể cần loại chỉ được **gắn cờ**, không bị xóa. Việc chọn dòng và biến cho
đề tài thuộc bước sau.

| Bước | Hàm | Việc làm | Ví dụ |
|---|---|---|---|
| 1 | `read_raw` | Đọc mọi cột dạng chuỗi, xóa khoảng trắng, đổi ô trống, `"-"`, `"Unknown Release Date"` thành giá trị thiếu; gán `gpu_id` | `"\n01-Mar-2009 "` → `"01-Mar-2009"` |
| 2 | `parse_columns` | Đổi từng cột sang đúng kiểu; **gặp định dạng lạ thì dừng và báo lỗi**; kiểm tra tính nhất quán giữa các cột | `"738 MHz"` → 738; `"800MB/sec"` → 0,8; `"24 (x4)"` → 96; `"450 Watt & 38 Amps"` → 450 và 38 |
| 3 | `add_flags` | Gắn cờ `is_multi_gpu` (theo luật tên sản phẩm), `is_unreleased`, `is_exact_duplicate` | 246 / 18 / 0 dòng |
| 4 | `describe_columns` | Lập từ điển dữ liệu: cột gốc, kiểu, số và % thiếu, ý nghĩa | |

**Đầu ra** (`data/interim/`):

| File | Dùng cho |
|---|---|
| `gpus_all_clean.parquet` | Code đọc: giữ nguyên kiểu dữ liệu. Dùng `clean.load_all_clean()` |
| `gpus_all_clean.csv` | Người xem: mở bằng Excel để kiểm tra |
| `gpus_all_clean_dictionary.csv` | Từ điển dữ liệu |


### Nhận diện card nhiều GPU

Cấu hình nhiều GPU (ghép nhiều card SLI/CrossFire, card hai chip trên một bo mạch, đồ
họa lai AMD Dual Graphics) được nhận diện **tự động** bằng 5 luật dựa trên tên sản phẩm
(`MULTI_GPU_RULES` trong `clean.py`):

| Luật | Dấu hiệu | Số dòng |
|---|---|---|
| `sli_crossfire` | "SLI", "CrossFire", "3-Way" | 180 |
| `dual_chip_model` | Danh sách mẫu card hai chip đã biết: GTX 690/590/295, Titan Z, HD 5970/6990/7990/8990, Radeon Pro Duo, Devil 13, ARES, MARS, R9 295X2, Tesla K10, FirePro S10000, Quadro NVS 420/450 | 41 |
| `hybrid_graphics` | Tên có dấu "+" (APU + card rời) | 20 |
| `x2_board` | Số hiệu kết thúc bằng "X2", "GX2", "PowerColor X2 Edition" | 14 |
| `quadro_plex` | Quadro Plex (hệ đồ họa ngoài) | 1 |

Tổng cộng **246 dòng** (một dòng có thể khớp nhiều luật).

**Cách xây dựng luật:** mọi dòng khớp luật đã được **kiểm tra thủ công** 
bằng thông số trong chính dữ liệu: card hai chip có TMUs và độ rộng bus gấp đôi card
đơn tương ứng. Các luật được viết hẹp để tránh bắt nhầm card một GPU: "Quadro",
"Dual-X", "TurboDuo", tản nhiệt "IceQ X2" / "Twin X2" / "WindForce 2X", chip laptop
"HD 6990M", mẫu "iGame ARES X" của Colorful. Mỗi trường hợp này có test riêng trong
`tests/test_parsers.py`.

### Các điểm cần biết về dữ liệu

- `ROPs` và `L2_Cache` dạng `"a (xN)"` là **phép nhân**, không phải số GPU: Titan XP
  ghi `"24 (x4)"` và có 96 ROPs.
- **Ở 305 dòng có ROPs dạng `"a (xN)"`, cột `TMUs` và `Memory_Bus` bị ghi chia cho
  N** (ký hiệu `(xN)` chỉ còn ở ROPs). Bằng chứng: `Texture_Rate` = TMUs × xung nhịp
  và băng thông = bus × tốc độ bộ nhớ chỉ khớp sau khi nhân lại với N. Code nhân lại
  và lưu N ở cột `spec_multiplier`. Ví dụ GTX 1080 Ti: TMUs 56 → 224, bus 88 → 352 bit.
- Cùng một phiên bản nhưng viết khác nhau được gộp về một cách viết:
  `"DX 12"` và `"DX 12.0"` → `"12.0"`; Shader Model `"5"` và `"5.0"` → `"5.0"`.
- `Pixel_Rate`, `Texture_Rate` được tính theo **xung boost** khi card có boost, và được
  làm tròn tới số nguyên trong file gốc.
- `Memory_Speed` không so sánh được giữa các loại bộ nhớ: băng thông = bus/8 ×
  tốc độ × k, với k = 1 (DDR), 2 (DDR2/DDR3), 4 (GDDR5), 8 (GDDR5X).
- Script tự kiểm tra hai đẳng thức trên ở bước 2 và báo số dòng không khớp. Hiện còn
  4 dòng băng thông không khớp, đều là cấu hình SLI (băng thông của hai card cộng lại).
- Một tên sản phẩm bị lỗi mã hóa ký tự (`"XÂ²"`) được sửa thành `"X²"`.
- `Shader` là phiên bản Shader Model (ví dụ 5.0), **không phải** số nhân xử lý.
- `PSU` là bộ nguồn khuyến nghị cho **cả hệ thống**, không phải công suất của card.
- `Power_Connector = "None"` không đáng tin: nhiều card trên 75 W vẫn ghi "None".

## Bước 2: dữ liệu cho đề tài (`scripts/02_build_dataset.py`)

Áp dụng các **quyết định riêng của đề tài** lên kết quả bước 1.

| Bước | Hàm | Việc làm | Kết quả hiện tại |
|---|---|---|---|
| 1 | `filter_rows` | Giữ GPU rời → bỏ card nhiều GPU → bỏ card chưa phát hành → bỏ dòng trùng | 3406 → 3018 → 2775 → 2762 → 2762 dòng |
| 2 | `derive_variables` | Thêm `segment` (Desktop/Laptop), `needs_aux_power` (trên 75 W), `texture_rate_per_watt`, `memory_type_group` (gộp các loại có dưới 30 card **có số liệu công suất** thành "Other", để mỗi nhóm đủ lớn cho ANOVA); **chỉ giữ các cột có phân tích sử dụng** | 2318 desktop, 444 laptop |
| 3 | `describe_columns` | Từ điển dữ liệu, có cột `role` (vai trò) và `used_in` (phân tích sử dụng) | 16 cột |

**Đầu ra** (`data/processed/`): `gpus_analysis.parquet` (code đọc bằng
`dataset.load_analysis()`), `gpus_analysis.csv`, `gpus_analysis_dictionary.csv`,
`row_counts.csv` (số dòng sau từng bước lọc, dùng cho báo cáo).

**16 cột của dữ liệu cuối:**

| Vai trò (`role`) | Cột | Dùng cho |
|---|---|---|
| Định danh (`identifier`) | `gpu_id`, `name` | Mọi phân tích (nhận diện từng card) |
| Biến mục tiêu (`response`) | `max_power_w` | Mọi phân tích |
| Biến tạo từ mục tiêu (`derived_response`) | `needs_aux_power` (công suất trên 75 W) | Thống kê mô tả; khoảng tin cậy cho tỉ lệ card cần nguồn phụ |
| | `texture_rate_per_watt` (hiệu quả năng lượng) | Thống kê mô tả; thảo luận |
| Biến giải thích (`predictor`) | `tmus`, `core_speed_mhz`, `memory_bandwidth_gb_s`, `memory_mb`, `process_nm` | Thống kê mô tả; hồi quy; kiểm chứng ngoài mẫu |
| Biến phân nhóm (`grouping`) | `manufacturer`, `segment` | Thống kê mô tả; so sánh hai nhóm; ANOVA; hồi quy |
| | `memory_type` | Thống kê mô tả (bảng tần số trước khi gộp nhóm) |
| | `memory_type_group` | Thống kê mô tả; ANOVA |
| Thời gian (`time`) | `release_year` | Thống kê mô tả (xu hướng theo năm); kiểm chứng ngoài mẫu (chia giai đoạn) |
| Cụm (`cluster`) | `architecture` (các card cùng chip không độc lập với nhau) | Hồi quy (sai số chuẩn theo cụm) |

Các cột không phân tích nào dùng (`release_date`, `rops`, `pixel_rate_gpixel_s`,
`texture_rate_gtexel_s`, `memory_speed_mhz`, `memory_bus_bit`, `power_connector`) không
đưa vào dữ liệu cuối nhưng **vẫn còn trong dữ liệu sạch dùng chung của bước 1**
(`gpus_all_clean`). Cần thêm cột nào thì thêm vào `COLUMNS` trong `dataset.py`.

**Nguyên tắc:** không xóa dòng thiếu dữ liệu và không điền giá trị cho biến đáp ứng
`max_power_w` (thiếu ở 481 dòng). Mỗi bài toán tự chọn các dòng đủ biến.

**Phạm vi:** tiền xử lý chỉ gồm đọc dữ liệu, xử lý định dạng, xử lý dữ liệu khuyết,
lọc quan sát, chọn biến và tạo biến mới. Các biến mới chỉ gồm những biến mà nhiều mục
phải dùng **cùng một định nghĩa** (ngưỡng 75 W, cách gộp nhóm bộ nhớ). **Ngoại lai,
biến đổi log và mẫu riêng của từng mô hình không được xử lý ở bước này**; mỗi mục phân
tích tự làm trong bối cảnh của mình (ví dụ lấy log ngay trong công thức hồi quy).

## Dữ liệu này dùng được cho những phân tích nào

Phần này chỉ là **gợi ý**: nêu cột cần dùng, cách lọc và số dòng sẵn có. Việc chọn phương
pháp, xử lý ngoại lai và lấy log do từng phân tích tự quyết định. Số dòng dưới đây chỉ
tính các card **có số liệu công suất** (`max_power_w` không thiếu).

| Phân tích | Cột chính | Cách lọc | Dữ liệu sẵn có |
|---|---|---|---|
| Thống kê mô tả | `max_power_w` và các cột còn lại | Có công suất | 2281 GPU, năm 2000–2017; 2094 card có hiệu quả năng lượng |
| Khoảng tin cậy cho tỉ lệ card cần nguồn phụ | `needs_aux_power` | Desktop, có công suất | 1913 GPU, trong đó 1310 card trên 75 W |
| So sánh hai nhóm | `max_power_w` theo `segment` hoặc `manufacturer` | Có công suất; Nvidia và AMD khi so sánh hãng | Desktop 1913 / laptop 368; Nvidia 1341 / AMD 856 |
| ANOVA | `max_power_w` theo `memory_type_group`; hoặc theo `manufacturer` × `segment` | Có công suất; Nvidia và AMD cho ANOVA hai yếu tố | 6 nhóm bộ nhớ, nhóm nhỏ nhất 33 card; bảng hãng × phân khúc, ô nhỏ nhất 137 card |
| Hồi quy | `max_power_w` theo `tmus`, `core_speed_mhz`, `memory_bandwidth_gb_s`, `memory_mb`, `process_nm`, `manufacturer`; cụm theo `architecture` | Desktop, Nvidia và AMD, đủ 6 biến | 1557 card thuộc 155 kiến trúc chip; mọi biến đều dương nên lấy log được |
| Kiểm chứng ngoài mẫu theo thời gian | Như hồi quy, chia theo `release_year` | Như hồi quy | 1083 card đến 2014, 450 card 2015–2017 (24 card không rõ năm) |

Ví dụ lấy mẫu cho hồi quy:

```python
from gpu_power_analysis.dataset import load_analysis

gpus = load_analysis()
predictors = ["tmus", "core_speed_mhz", "memory_bandwidth_gb_s", "memory_mb", "process_nm"]
sample = gpus[
    gpus["segment"].eq("Desktop") & gpus["manufacturer"].isin(["Nvidia", "AMD"])
].dropna(subset=["max_power_w", *predictors])   # 1557 rows
```

## Quy chuẩn code

- Toàn bộ code bằng tiếng Anh (tên, comment, docstring, commit message); tiếng Việt chỉ
  dùng trong tài liệu và báo cáo.
- PEP 8, type hint, docstring kiểu Google; chạy `ruff` và `pytest` trước khi commit.

## Nguồn dữ liệu

Kaggle, *Computer Parts (CPUs and GPUs)*: https://www.kaggle.com/datasets/iliassekkaf/computerparts
