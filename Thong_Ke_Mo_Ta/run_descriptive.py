#python Thong_Ke_Mo_Ta/run_descriptive.py
import sys
import io
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

CURRENT_DIR = Path(__file__).resolve().parent
DATA_PATH = CURRENT_DIR.parent / "data" / "processed" / "gpus_analysis.parquet"

# Đọc thẳng file parquet và xóa danh mục rỗng (ẩn Intel vì có 0 mẫu)
df = pd.read_parquet(DATA_PATH)
df['manufacturer'] = df['manufacturer'].cat.remove_unused_categories()

# 1. THỐNG KÊ ĐẶC TRƯNG BIẾN ĐỊNH LƯỢNG (Đã bỏ core_speed_mhz)
num_cols = [
    'max_power_w', 
    'process_nm', 
    'memory_bandwidth_gb_s', 
    'memory_mb', 
    'tmus',
    'texture_rate_per_watt'
]

stats_df = pd.DataFrame({
    'Min': df[num_cols].min(),
    'Max': df[num_cols].max(),
    'Mean': df[num_cols].mean(),
    'SD': df[num_cols].std(),
    'Median': df[num_cols].median(),
    'Q1': df[num_cols].quantile(0.25),
    'Q3': df[num_cols].quantile(0.75),
    'IQR': df[num_cols].quantile(0.75) - df[num_cols].quantile(0.25),
}).round(2)

print("=== BẢNG THỐNG KÊ ĐẶC TRƯNG MẪU ===")
print(stats_df.to_string())
stats_df.to_csv(CURRENT_DIR / "bang_thong_ke_mo_ta.csv")

# 2. THỐNG KÊ TẦN SỐ & TẦN SUẤT (BẢNG 2)
n_total = len(df)
freq_df = df['manufacturer'].value_counts().reset_index()
freq_df.columns = ['Hãng sản xuất', 'Số lượng mẫu (Tần số)']
freq_df['Tỷ lệ (%)'] = (freq_df['Số lượng mẫu (Tần số)'] / n_total * 100).round(2)
freq_df['Tỷ lệ tích lũy (%)'] = freq_df['Tỷ lệ (%)'].cumsum().round(2)

print("\n=== BẢNG 2: THỐNG KÊ TẦN SỐ & TẦN SUẤT HÃNG SẢN XUẤT ===")
print(freq_df.to_string(index=False))
print(f"Tổng cộng: {n_total} quan trắc (100.00%)")
freq_df.to_csv(CURRENT_DIR / "bang_tan_so_hang.csv", index=False)

# 3. VẼ VÀ LƯU ĐỒ THỊ (.PNG) VÀO CÙNG FOLDER
sns.set_theme(style="whitegrid")

# Graph 1: Histogram Max_Power (Gốc vs Log)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.histplot(df['max_power_w'], kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Phân bố Max_Power (Watts)')

sns.histplot(np.log(df['max_power_w']), kde=True, ax=axes[1], color='salmon')
axes[1].set_title('Phân bố ln(Max_Power)')
plt.tight_layout()
plt.savefig(CURRENT_DIR / "histogram_max_power.png", dpi=300)
plt.close()

# Graph 2: Boxplot Desktop vs Laptop (Không còn cột trống của Intel)
plt.figure(figsize=(8, 4.5))
sns.boxplot(x='manufacturer', y='max_power_w', hue='segment', data=df, palette='Set2')
plt.title('So sánh Max_Power giữa Desktop và Laptop')
plt.tight_layout()
plt.savefig(CURRENT_DIR / "boxplot_max_power.png", dpi=300)
plt.close()

# Graph 3: Correlation Matrix Heatmap (6 biến)
plt.figure(figsize=(7, 5))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Ma trận hệ số tương quan Pearson')
plt.tight_layout()
plt.savefig(CURRENT_DIR / "correlation_matrix.png", dpi=300)
plt.close()

print("\n===> THÀNH CÔNG: Đã lưu bảng .csv và 3 file đồ thị .png ngay trong thư mục này!")