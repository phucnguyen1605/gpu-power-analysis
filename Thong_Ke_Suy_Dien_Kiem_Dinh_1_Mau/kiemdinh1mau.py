# python Thong_Ke_Suy_Dien_Kiem_Dinh_1_Mau/kiemdinh1mau.py
import sys
import io
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

CURRENT_DIR = Path(__file__).resolve().parent
DATA_PATH = CURRENT_DIR.parent / "data" / "processed" / "gpus_analysis.parquet"

# Đọc dữ liệu sạch
df = pd.read_parquet(DATA_PATH)
power = df['max_power_w'].dropna().astype(float)

# 1. THỐNG KÊ ĐẶC TRƯNG MẪU & KIỂM TRA PHÂN PHỐI CHUẨN (BẢNG 1)
n = len(power)
mean_val = power.mean()
sd_val = power.std(ddof=1)
se_val = sd_val / np.sqrt(n)
median_val = power.median()
iqr_val = power.quantile(0.75) - power.quantile(0.25)
w_shapiro, p_shapiro = stats.shapiro(power)

sample_df = pd.DataFrame([{
    'Cỡ mẫu (n)': n,
    'Mean (W)': round(mean_val, 2),
    'SD (W)': round(sd_val, 2),
    'SE (W)': round(se_val, 2),
    'Median (W)': round(median_val, 2),
    'IQR (W)': round(iqr_val, 2),
    'Shapiro W': round(w_shapiro, 3),
    'Shapiro p': f"{p_shapiro:.2e}"
}])

print("=== BẢNG 1: ĐẶC TRƯNG MẪU & KIỂM TRA PHÂN PHỐI CHUẨN ===")
print(sample_df.to_string(index=False))
sample_df.to_csv(CURRENT_DIR / "bang_dac_trung_1_mau.csv", index=False, encoding='utf-8-sig')

# 2. KIỂM ĐỊNH TRUNG BÌNH MỘT MẪU & KIỂM TRA ĐỘ BỀN VỮNG (BẢNG 2)
mu_0 = 75.0
df_t = n - 1

# Kiểm định t một mẫu (H1: mu > 75) & KTC 95%
t_stat, p_two = stats.ttest_1samp(power, popmean=mu_0)
p_right = p_two / 2
t_crit_right = stats.t.ppf(0.95, df=df_t)
t_crit_two = stats.t.ppf(0.975, df=df_t)
ci_low = mean_val - t_crit_two * se_val
ci_high = mean_val + t_crit_two * se_val

# Kiểm định phi tham số Wilcoxon có dấu
w_wilcox, p_wilcox = stats.wilcoxon(power - mu_0, alternative='greater')

# Kiểm định t trên thang đo logarit ln(P)
log_p = np.log(power)
t_log, p_log = stats.ttest_1samp(log_p, popmean=np.log(mu_0), alternative='greater')
gmean = np.exp(log_p.mean())
log_se = log_p.std(ddof=1) / np.sqrt(n)
gmean_low = np.exp(log_p.mean() - t_crit_two * log_se)
gmean_high = np.exp(log_p.mean() + t_crit_two * log_se)

test_df = pd.DataFrame([
    {
        'Phương pháp': 'Kiểm định t một mẫu (W)',
        'Thống kê': f"t = {t_stat:.2f} (df = {df_t})",
        'Giá trị tới hạn': f"t_0.95 = {t_crit_right:.2f}; t_0.975 = {t_crit_two:.2f}",
        'Ước lượng': f"{mean_val:.2f} W",
        'KTC 95%': f"[{ci_low:.2f}; {ci_high:.2f}] W",
        'Giá trị p': f"{p_right:.2e}"
    },
    {
        'Phương pháp': 'Wilcoxon có dấu (W)',
        'Thống kê': f"W = {w_wilcox:,.0f}",
        'Giá trị tới hạn': '--',
        'Ước lượng': f"Trung vị {median_val:.2f} W",
        'KTC 95%': '--',
        'Giá trị p': f"{p_wilcox:.2e}"
    },
    {
        'Phương pháp': 'Kiểm định t trên ln(P)',
        'Thống kê': f"t = {t_log:.2f} (df = {df_t})",
        'Giá trị tới hạn': f"t_0.95 = {t_crit_right:.2f}",
        'Ước lượng': f"TB nhân {gmean:.2f} W",
        'KTC 95%': f"[{gmean_low:.2f}; {gmean_high:.2f}] W",
        'Giá trị p': f"{p_log:.2e}"
    }
])

print("\n=== BẢNG 2: TỔNG HỢP KẾT QUẢ KIỂM ĐỊNH MỘT MẪU (SO VỚI 75 W) ===")
print(test_df.to_string(index=False))
test_df.to_csv(CURRENT_DIR / "bang_kiem_dinh_1_mau.csv", index=False, encoding='utf-8-sig')

# 3. VẼ VÀ LƯU ĐỒ THỊ Q-Q VÀO CÙNG FOLDER
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(6, 4))
stats.probplot(power, plot=ax)
ax.set_title('Đồ thị Q-Q chuẩn của công suất tối đa (max_power_w)', fontsize=11)
ax.set_xlabel('Phân vị chuẩn')
ax.set_ylabel('Phân vị mẫu (Watts)')
ax.get_lines()[0].set_markersize(2.5)
plt.tight_layout()
plt.savefig(CURRENT_DIR / "one_sample_qq.pdf")
plt.savefig(CURRENT_DIR / "one_sample_qq.png", dpi=300)
plt.close()

print("\n===> THÀNH CÔNG: Đã lưu 2 bảng .csv và đồ thị Q-Q (.pdf, .png) ngay trong thư mục này!")