# src/verify_preprocessing.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

def verify_pipeline():
    print("=" * 60)
    print("WEEK 2: PREPROCESSING & SCALER VERIFICATION")
    print("=" * 60)

    train_path = "data/processed/train_scaled.csv"
    val_path = "data/processed/val_scaled.csv"
    test_path = "data/processed/test_scaled.csv"

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing processed file: {p}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"Train Scaled Shape:      {train_df.shape}")
    print(f"Validation Scaled Shape: {val_df.shape}")
    print(f"Test Scaled Shape:       {test_df.shape}")

    train_min = train_df.min().min()
    train_max = train_df.max().max()
    print(f"\nTrain set value bounds: Min = {train_min:.4f}, Max = {train_max:.4f}")
    assert np.isclose(train_min, 0.0, atol=1e-3), "Train min is not 0"
    assert np.isclose(train_max, 1.0, atol=1e-3), "Train max is not 1"
    print("Train set [0, 1] scaling bounds strictly validated.")

    scaler_X = joblib.load("models/scaler_X.pkl")
    scaler_y = joblib.load("models/scaler_y.pkl")

    raw_df = pd.read_csv("data/raw/engine_dataset.csv")
    sample_raw_X = raw_df.iloc[:5, :7].values
    scaled_X = scaler_X.transform(sample_raw_X)
    recovered_X = scaler_X.inverse_transform(scaled_X)

    diff = np.max(np.abs(sample_raw_X - recovered_X))
    print(f"Maximum inverse-transform reconstruction error: {diff:.2e}")
    assert diff < 1e-5, "Scaler reconstruction error is too high!"
    print("Scaler persistence and bidirectional transformation verified.")

    os.makedirs("reports/figures", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].hist(raw_df["engine_speed"], bins=20, color="steelblue", edgecolor="black")
    axes[0, 0].set_title("Raw Engine Speed (rpm)")
    axes[0, 0].set_xlabel("rpm")
    axes[0, 0].set_ylabel("Count")

    axes[0, 1].hist(train_df["in_engine_speed"], bins=20, color="orange", edgecolor="black")
    axes[0, 1].set_title("Normalized Engine Speed [0, 1]")
    axes[0, 1].set_xlabel("Normalized Value")

    axes[1, 0].hist(raw_df["bsfc"], bins=20, color="forestgreen", edgecolor="black")
    axes[1, 0].set_title("Raw BSFC (g/kWh)")
    axes[1, 0].set_xlabel("g/kWh")
    axes[1, 0].set_ylabel("Count")

    axes[1, 1].hist(train_df["out_bsfc"], bins=20, color="coral", edgecolor="black")
    axes[1, 1].set_title("Normalized BSFC [0, 1]")
    axes[1, 1].set_xlabel("Normalized Value")

    plt.tight_layout()
    plot_path = "reports/figures/week2_scaling_verification.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\nGenerated scaling visual deliverable: {plot_path}")
    print("=" * 60)
    print("WEEK 2 PREPROCESSING VERIFICATION COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    verify_pipeline()