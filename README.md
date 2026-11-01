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
python scripts/01_preprocess.py                  # tiền xử lý: dữ liệu thô -> dữ liệu cho đề tài
python scripts/reports/report_preprocessing.py   # số liệu cho mục tiền xử lý của báo cáo (LaTeX)
(cd report && tectonic preview_section3.tex)     # xem trước mục 3 (cần cài Tectonic)
pytest                                           # chạy test
ruff format . && ruff check .                    # format và kiểm tra code trước khi commit
```

Dữ liệu gốc `data/raw/All_GPUs.csv` có sẵn trong repo, không cần tải từ Kaggle. Các file
sinh ra (`data/processed/`, `report/generated/`, `report/preview_section3.pdf`) cũng được
lưu trong git để cả nhóm dùng ngay mà không cần chạy code. **Khi sửa code, chạy lại các
lệnh trên và commit các file sinh ra cùng với code.**

## Cấu trúc

```
gpu-power-analysis/
├── data/
│   ├── raw/All_GPUs.csv                 dữ liệu gốc, không bao giờ sửa
│   └── processed/                       dữ liệu cho đề tài (do script sinh ra)
├── src/gpu_power_analysis/
│   ├── paths.py                         đường dẫn tới mọi file
│   ├── parsers.py                       các hàm đổi chuỗi thành số, ngày, Yes/No
│   └── preprocess.py                    6 bước tiền xử lý, mỗi bước một hàm
├── scripts/
│   ├── 01_preprocess.py                 chạy tiền xử lý, in kết quả từng bước
│   └── reports/report_preprocessing.py  sinh số liệu cho mục 3 của báo cáo
├── report/
│   ├── sections/03_preprocessing.tex    mục 3 (Tiền xử lý số liệu)
│   ├── generated/                       macro số liệu và bảng mẫu (do script sinh ra)
│   └── preview_section3.tex             file xem trước mục 3
├── tests/test_parsers.py                test cho các hàm đọc giá trị
└── tests/test_preprocess.py             test cho quy trình tiền xử lý
```

## Tiền xử lý (`scripts/01_preprocess.py`)

| Bước | Hàm | Việc làm | Kết quả hiện tại |
|---|---|---|---|
| 1 | `read_raw` | Đọc mọi cột dưới dạng chuỗi, chưa đổi giá trị nào; gán `gpu_id` | 3406 dòng × 34 cột |
| 2 | `select_columns` | Chỉ giữ 18 cột cần cho đề tài (`USED_COLUMNS`) | 18 cột |
| 3 | `clean_text` | Xóa khoảng trắng; đổi ô trống, `"-"`, `"Unknown Release Date"` thành giá trị thiếu; sửa lỗi mã hóa tên | `"\n01-Mar-2009 "` → `"01-Mar-2009"` |
| 4 | `parse_columns` | Tách đơn vị, đổi kiểu, sửa lỗi `(xN)`; **gặp định dạng lạ thì dừng và báo lỗi**; `check_consistency` kiểm tra kết quả | `"738 MHz"` → 738; `"800MB/sec"` → 0,8; TMUs `56` → 224 khi ROPs ghi `"24 (x4)"` |
| 5 | `filter_rows` | Giữ GPU rời → bỏ cấu hình nhiều GPU → bỏ card chưa phát hành → bỏ dòng trùng | 3406 → 3018 → 2775 → 2762 → 2762 dòng |
| 6 | `derive_variables` | Thêm `segment`, `needs_aux_power` (trên 75 W, chỉ có ý nghĩa với card desktop), `texture_rate_per_watt`, `memory_type_group` (gộp các loại có dưới 30 card **có số liệu công suất** thành "Other"); giữ 16 cột cuối | 2318 desktop, 444 laptop |

Sau 6 bước, `describe_columns` tạo từ điển dữ liệu (cột `role`: vai trò, `used_in`: phân
tích sử dụng) rồi script ghi các file đầu ra.

**Đầu ra** (`data/processed/`):

| File | Dùng cho |
|---|---|
| `gpus_analysis.parquet` | Code đọc: giữ nguyên kiểu dữ liệu. Dùng `preprocess.load_analysis()` |
| `gpus_analysis.csv` | Người xem: mở bằng bảng tính để kiểm tra |
| `gpus_analysis_dictionary.csv` | Từ điển dữ liệu |
| `row_counts.csv` | Số dòng sau từng bước lọc |

### Các cột được chọn (bước 2)

- **Có trong dữ liệu cuối:** `Name`, `Manufacturer`, `Architecture`, `Release_Date`,
  `Notebook_GPU`, `Max_Power`, `TMUs`, `Core_Speed`, `Memory`, `Memory_Bandwidth`,
  `Memory_Type`, `Process`.
- **Chỉ dùng trong lúc xử lý:** `Dedicated` (lọc GPU rời), `ROPs` (lấy hệ số N để sửa
  lỗi `(xN)`), `Texture_Rate` (kiểm tra lỗi, tính hiệu quả năng lượng), `Boost_Clock`,
  `Memory_Bus`, `Memory_Speed` (kiểm tra lỗi).
- 16 cột còn lại của dữ liệu gốc không được xử lý:
  - cổng xuất hình, độ phân giải, phiên bản DirectX/OpenGL, giá bán: không liên quan tới
    đề tài;
  - `PSU`, `Power_Connector`: được chọn **sau khi** đã biết công suất, đưa vào mô hình là
    rò rỉ thông tin của biến mục tiêu;
  - `Pixel_Rate`: bằng ROPs × xung nhịp, trùng thông tin với `TMUs` và `Core_Speed`;
  - `Shader`: là phiên bản shader (như 5.0), không phải số nhân xử lý;
  - `L2_Cache`: nhiều card ghi "0KB", nhiều khả năng là không có số liệu;
  - `Integrated`: trùng thông tin với `Dedicated`;
  - `SLI_Crossfire`: chỉ cho biết card *hỗ trợ* ghép nhiều card.

  Cần thêm cột nào thì thêm vào `USED_COLUMNS`, viết dòng chuyển đổi trong
  `parse_columns` và khai báo trong `COLUMNS`.

### Nhận diện cấu hình nhiều GPU (bước 5)

Cấu hình nhiều GPU (ghép nhiều card SLI/CrossFire, card hai chip trên một bo mạch, đồ
họa lai AMD Dual Graphics) được nhận diện **tự động** bằng 5 luật dựa trên tên sản phẩm
(`MULTI_GPU_RULES` trong `preprocess.py`):

| Luật | Dấu hiệu | Số dòng |
|---|---|---|
| `sli_crossfire` | "SLI", "CrossFire", "2-Way"/"3-Way"/"4-Way" | 180 |
| `dual_chip_model` | Danh sách mẫu card hai chip đã biết: GTX 690/590/295, Titan Z, HD 5970/6990/7990/8990, Radeon Pro Duo, Devil 13, ARES, MARS, R9 295X2, Tesla K10, FirePro S10000, Quadro NVS 420/450 | 41 |
| `hybrid_graphics` | Tên có dấu "+" (APU + card rời) | 20 |
| `x2_board` | Số hiệu kết thúc bằng "X2", "GX2", "PowerColor X2 Edition" | 14 |
| `quadro_plex` | Quadro Plex (hệ đồ họa ngoài) | 1 |

Tổng cộng **246 dòng** trong dữ liệu gốc (một dòng có thể khớp nhiều luật). Bước 5 loại
**243 dòng**, vì 3 dòng đã bị loại trước đó do là GPU tích hợp.

**Cách xây dựng luật:** mọi dòng khớp luật đã được **kiểm tra thủ công**
bằng thông số trong chính dữ liệu: card hai chip có TMUs và độ rộng bus gấp đôi card
đơn tương ứng. Các luật được viết hẹp để tránh bắt nhầm card một GPU: "Quadro",
"Dual-X", "TurboDuo", tản nhiệt "IceQ X2" / "Twin X2" / "WindForce 2X", chip laptop
"HD 6990M", mẫu "iGame ARES X" của Colorful. Mỗi trường hợp này có test riêng trong
`tests/test_parsers.py`.

### Các điểm cần biết về dữ liệu

- `ROPs` dạng `"a (xN)"` là **phép nhân**, không phải số GPU: Titan XP ghi `"24 (x4)"`
  và có 96 ROPs.
- **Ở 305 dòng có ROPs dạng `"a (xN)"`, cột `TMUs` và `Memory_Bus` bị ghi chia cho
  N** (ký hiệu `(xN)` chỉ còn ở ROPs). Bằng chứng: `Texture_Rate` = TMUs × xung nhịp
  và băng thông = bus × tốc độ bộ nhớ chỉ khớp sau khi nhân lại với N. Code nhân lại.
  Ví dụ GTX 1080 Ti Zotac Founders Edition (ROPs `"22 (x4)"`): TMUs 56 → 224, bus 88 →
  352 bit, khớp với dòng GeForce GTX 1080 Ti bản tham chiếu.
- `Texture_Rate` được tính theo **xung boost** khi card có boost, và được làm tròn tới số
  nguyên trong file gốc.
- `Memory_Speed` không so sánh được giữa các loại bộ nhớ: băng thông = bus/8 ×
  tốc độ × k, với k = 1 (DDR), 2 (DDR2/DDR3), 4 (GDDR5), 8 (GDDR5X).
- Script tự kiểm tra hai đẳng thức trên ở bước 4 (một dòng không khớp nếu hai vế lệch
  nhau quá 10%) và báo số dòng không khớp. Hiện còn
  4 dòng băng thông không khớp, đều là cấu hình SLI (băng thông của hai card cộng lại),
  bị loại ở bước 5.
- Một tên sản phẩm bị lỗi mã hóa ký tự (`"XÂ²"`) được sửa thành `"X²"`.

### 16 cột của dữ liệu cuối

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

### Nguyên tắc và phạm vi

- **Không điền giá trị thiếu** cho biến mục tiêu `max_power_w` (thiếu ở 481 dòng). Mỗi
  bài toán tự chọn các dòng đủ biến.
- **14 dòng không rõ GPU rời hay tích hợp bị loại** theo quy tắc `Dedicated = Yes`,
  không điền tay. Không dòng nào trong số đó đủ biến cho mô hình hồi quy.
- **Không sửa tay giá trị** theo nguồn bên ngoài; chỉ sửa các lỗi phát hiện được một cách
  có hệ thống từ chính dữ liệu (lỗi `(xN)`).
- Tiền xử lý chỉ tạo những biến mới mà nhiều phân tích phải dùng **cùng một định nghĩa**
  (ngưỡng 75 W, cách gộp nhóm bộ nhớ). **Ngoại lai, biến đổi log và mẫu riêng của từng
  mô hình** do mỗi phân tích tự làm (ví dụ lấy log ngay trong công thức hồi quy).

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
from gpu_power_analysis.preprocess import load_analysis

gpus = load_analysis()
predictors = [
    "tmus",
    "core_speed_mhz",
    "memory_bandwidth_gb_s",
    "memory_mb",
    "process_nm",
]
sample = gpus[
    gpus["segment"].eq("Desktop") & gpus["manufacturer"].isin(["Nvidia", "AMD"])
].dropna(subset=["max_power_w", *predictors])  # 1557 rows
```

## Quy chuẩn code

- Toàn bộ code bằng tiếng Anh (tên, comment, docstring, commit message); tiếng Việt chỉ
  dùng trong tài liệu và báo cáo.
- PEP 8, type hint, docstring kiểu Google; chạy `ruff` và `pytest` trước khi commit.

## Nguồn dữ liệu

Kaggle, *Computer Parts (CPUs and GPUs)*: https://www.kaggle.com/datasets/iliassekkaf/computerparts
