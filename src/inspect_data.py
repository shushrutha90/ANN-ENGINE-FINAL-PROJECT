import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_path = "data/raw/engine_dataset.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Missing {data_path}. Run generate_dataset.py first.")

df = pd.read_csv(data_path)

print("=" * 60)
print("WEEK 1: ENGINE DATASET OVERVIEW (7 INPUTS, 8 OUTPUTS)")
print("=" * 60)
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

inputs = [
    "injection_timing", "injection_pressure", "air_fuel_ratio",
    "compression_ratio", "egr_rate", "engine_speed", "fuel_blend"
]
outputs = [
    "brake_power", "brake_thermal_efficiency", "bsfc",
    "nox", "co", "hc", "smoke_opacity", "co2"
]

print(f"Inputs  ({len(inputs)}): {inputs}")
print(f"Outputs ({len(outputs)}): {outputs}")

print("\n--- Missing Values Check ---")
print(f"Total nulls in dataset: {df.isnull().sum().sum()}")

print("\n--- Summary Statistics ---")
print(df.describe().T[["min", "mean", "max"]])

os.makedirs("reports/figures", exist_ok=True)

plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix: 7 Inputs vs 8 Outputs", fontsize=14, pad=12)
plt.tight_layout()
heatmap_path = "reports/figures/week1_correlation_matrix.png"
plt.savefig(heatmap_path, dpi=300)
plt.close()
print(f"\nUpdated correlation plot: {heatmap_path}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.scatterplot(data=df, x="egr_rate", y="nox", ax=axes[0], color="crimson", alpha=0.7)
axes[0].set_title("EGR Rate vs NOx (Emissions Trade-off)")
axes[0].set_xlabel("EGR Rate (%)")
axes[0].set_ylabel("NOx (g/kWh)")

sns.scatterplot(data=df, x="compression_ratio", y="brake_thermal_efficiency", ax=axes[1], color="teal", alpha=0.7)
axes[1].set_title("Compression Ratio vs BTE (Performance Trend)")
axes[1].set_xlabel("Compression Ratio")
axes[1].set_ylabel("Brake Thermal Efficiency (%)")

plt.tight_layout()
trends_path = "reports/figures/week1_key_trends.png"
plt.savefig(trends_path, dpi=300)
plt.close()
print(f"Updated combustion trends plot: {trends_path}")
print("=" * 60)
print("WEEK 1 VERIFICATION COMPLETED (15 COLUMNS: 7 IN / 8 OUT)")
print("=" * 60)