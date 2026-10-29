# GPU Power Analysis

Mã nguồn cho Bài tập lớn môn Xác suất và Thống kê (MT2013), HCMUT:

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
python scripts/01_clean_raw.py       # làm sạch dữ liệu thô
pytest                               # chạy test
ruff format . && ruff check .        # format và kiểm tra code trước khi commit
```

## Cấu trúc

```
gpu-power-analysis/
├── data/
│   ├── raw/All_GPUs.csv                 dữ liệu gốc, không bao giờ sửa
│   ├── manual/multi_gpu_review.csv      quyết định rà soát tay (đưa lên git)
│   └── interim/                         kết quả do script sinh ra (không đưa lên git)
├── src/gpu_power_analysis/
│   ├── paths.py                         đường dẫn tới mọi file
│   ├── parsers.py                       các hàm đổi chuỗi thành số, ngày, Yes/No
│   └── clean.py                         6 bước làm sạch, mỗi bước một hàm
├── scripts/01_clean_raw.py              chạy 6 bước, in kết quả từng bước
├── tests/test_parsers.py                test cho các hàm đọc giá trị
└── tests/test_clean_invariants.py       test file sạch không mất hay tự sinh dữ liệu
```

## Bước làm sạch (`scripts/01_clean_raw.py`)

Kết quả **giữ đủ 3406 dòng và 34 biến gốc**. Bước này chỉ làm những việc khách quan;
các dòng có thể cần loại chỉ được **gắn cờ**, không bị xóa. Việc chọn dòng và biến cho
đề tài thuộc bước sau.

| Bước | Hàm | Việc làm | Ví dụ |
|---|---|---|---|
| 1 | `read_raw` | Đọc mọi cột dạng chuỗi, xóa khoảng trắng, đổi ô trống, `"-"`, `"Unknown Release Date"` thành giá trị thiếu; gán `gpu_id` | `"\n01-Mar-2009 "` → `"01-Mar-2009"` |
| 2 | `find_multi_gpu_candidates` | Dò tên theo 5 luật để tìm card nhiều GPU | 241 ứng viên |
| 3 | `apply_review` | Lấy quyết định cuối từ file rà soát tay | `exclude=False` cho card bắt nhầm |
| 4 | `parse_columns` | Đổi từng cột sang đúng kiểu; **gặp định dạng lạ thì dừng và báo lỗi** | `"738 MHz"` → 738; `"800MB/sec"` → 0,8; `"24 (x4)"` → 96; `"450 Watt & 38 Amps"` → 450 và 38 |
| 5 | `add_flags` | Gắn cờ `is_multi_gpu`, `is_unreleased`, `is_exact_duplicate` | 241 / 18 / 0 dòng |
| 6 | `describe_columns` | Lập từ điển dữ liệu: cột gốc, kiểu, số và % thiếu, ý nghĩa | |

**Đầu ra** (`data/interim/`):

| File | Dùng cho |
|---|---|
| `gpus_all_clean.parquet` | Code đọc: giữ nguyên kiểu dữ liệu. Dùng `clean.load_all_clean()` |
| `gpus_all_clean.csv` | Người xem: mở bằng Excel để kiểm tra |
| `gpus_all_clean_dictionary.csv` | Từ điển dữ liệu |
| `multi_gpu_candidates.csv` | Danh sách ứng viên nhiều GPU trước khi rà tay |

**Đã kiểm chứng (29/10/2026):** toàn bộ 149.864 ô của file sạch được đối chiếu với file
thô bằng một cách tính độc lập (chỉ dùng thao tác chuỗi, không dùng `parsers.py`):
0 khác biệt. Script đối chiếu đã được gỡ khỏi repo để tránh phải bảo trì logic hai lần;
`tests/test_clean_invariants.py` giữ lại phần kiểm tra không mất dữ liệu.

### Rà soát card nhiều GPU

Lần chạy đầu tạo `data/manual/multi_gpu_review.csv`. Với mỗi dòng:

- giữ `exclude=True` nếu đúng là nhiều GPU (SLI, CrossFire, card hai chip, đồ họa lai);
- đổi `exclude=False` nếu là card một GPU bị bắt nhầm;
- đặt `reviewed=True` và ghi lý do vào `note`.

Có thể thêm dòng mới cho card nhiều GPU mà luật không tìm ra. Chạy lại script sau khi
rà; script cảnh báo cho đến khi mọi dòng đã được rà.

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
- Script tự kiểm tra hai đẳng thức trên ở bước 4 và báo số dòng không khớp. Hiện còn
  4 dòng băng thông không khớp, đều là cấu hình SLI (băng thông của hai card cộng lại).
- Một tên sản phẩm bị lỗi mã hóa ký tự (`"XÂ²"`) được sửa thành `"X²"`.
- `Shader` là phiên bản Shader Model (ví dụ 5.0), **không phải** số nhân xử lý.
- `PSU` là bộ nguồn khuyến nghị cho **cả hệ thống**, không phải công suất của card.
- `Power_Connector = "None"` không đáng tin: nhiều card trên 75 W vẫn ghi "None".

## Quy chuẩn code

- Toàn bộ code bằng tiếng Anh (tên, comment, docstring, commit message); tiếng Việt chỉ
  dùng trong tài liệu và báo cáo.
- PEP 8, type hint, docstring kiểu Google; chạy `ruff` và `pytest` trước khi commit.

## Nguồn dữ liệu

Kaggle, *Computer Parts (CPUs and GPUs)*: https://www.kaggle.com/datasets/iliassekkaf/computerparts
