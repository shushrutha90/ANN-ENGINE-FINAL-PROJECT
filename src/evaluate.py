import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score, root_mean_squared_error
from model import EngineANN

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def evaluate():
    test_df = pd.read_csv("data/processed/test_scaled.csv")
    X_test_scaled = torch.tensor(test_df.iloc[:, :7].values, dtype=torch.float32)
    y_test_scaled = test_df.iloc[:, 7:].values

    model = EngineANN(input_dim=7, output_dim=8)
    model.load_state_dict(torch.load("models/best_ann_model.pth"))
    model.eval()

    with torch.no_grad():
        preds_scaled = model(X_test_scaled).numpy()

    scaler_y = joblib.load("models/scaler_y.pkl")
    preds_raw = scaler_y.inverse_transform(preds_scaled)
    actual_raw = pd.read_csv("data/processed/y_test_raw.csv").values

    target_names = scaler_y.feature_names_in_

    metrics = []
    for i, target in enumerate(target_names):
        r2 = r2_score(actual_raw[:, i], preds_raw[:, i])
        rmse = root_mean_squared_error(actual_raw[:, i], preds_raw[:, i])
        mape = calculate_mape(actual_raw[:, i], preds_raw[:, i])
        metrics.append({
            "Target": target,
            "R2": round(r2, 4),
            "RMSE": round(rmse, 4),
            "MAPE (%)": round(mape, 2)
        })

    metrics_df = pd.DataFrame(metrics)
    print("=" * 60)
    print("WEEK 3: TEST SET PERFORMANCE EVALUATION (UNSEEN DATA)")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    print("=" * 60)

    os.makedirs("reports", exist_ok=True)
    metrics_df.to_csv("reports/week3_test_metrics.csv", index=False)

    os.makedirs("reports/figures", exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for i, target in enumerate(target_names):
        ax = axes[i]
        ax.scatter(actual_raw[:, i], preds_raw[:, i], alpha=0.75, color="royalblue", edgecolors="k", s=30)
        min_val = min(actual_raw[:, i].min(), preds_raw[:, i].min())
        max_val = max(actual_raw[:, i].max(), preds_raw[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=1.5)
        ax.set_title(f"{target}\n$R^2$: {metrics[i]['R2']}", fontsize=10)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = "reports/figures/week3_actual_vs_predicted.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved parity scatter plots to: {plot_path}")

if __name__ == "__main__":
    evaluate()