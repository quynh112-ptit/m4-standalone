import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/synthetic_ids.csv")

print("=== BƯỚC 1: KIỂM TRA SỐ DÒNG/CỘT VÀ DỮ LIỆU LỖI (NaN/Inf) ===")
print(df.info())
print("Số giá trị bị trống (NaN):", df.isna().sum().sum())
print("Số giá trị vô cực (Inf):", np.isinf(df.select_dtypes(include=np.number)).sum().sum())

print("\n=== BƯỚC 2: SỐ MẪU MỖI LỚP ===")
print(df["label"].value_counts())

print("\n=== BƯỚC 3: THỐNG KÊ MIN / MEDIAN(50%) / MEAN / MAX CỦA 4 FEATURE ===")
cols = ["duration_s", "total_packets", "byte_rate", "dst_port"]
print(df.groupby("label")[cols].describe())

# Bước 4: Vẽ histogram total_packets và byte_rate theo lớp và lưu ra file ảnh
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for label, grp in df.groupby("label"):
    axes[0].hist(grp["total_packets"], bins=30, alpha=0.6, label=label)
    axes[1].hist(grp["byte_rate"], bins=30, alpha=0.6, label=label)

axes[0].set_title("Histogram: total_packets (SIMULATED DATA)")
axes[0].set_xlabel("total_packets")
axes[0].legend()

axes[1].set_title("Histogram: byte_rate (SIMULATED DATA)")
axes[1].set_xlabel("byte_rate")
axes[1].legend()

plt.tight_layout()
plt.savefig("results/lab4_1_histogram.png")
print("\n=== BƯỚC 4 & 5: ĐÃ LƯU BIỂU ĐỒ TẠI results/lab4_1_histogram.png ===")
print("LƯU Ý: Đây là phân bố mô phỏng (SIMULATED DATA - NOT REAL NETWORK IDS PERFORMANCE)")
print("PASS Lab 4.1!")
