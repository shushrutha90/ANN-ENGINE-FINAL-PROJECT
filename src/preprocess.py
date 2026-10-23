import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

def run_preprocessing(raw_data_path="data/raw/engine_dataset.csv", output_dir="data/processed"):
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_data_path}")

    df = pd.read_csv(raw_data_path)
    print(f"Loaded raw dataset with shape: {df.shape}")

    feature_cols = [
        "engine_speed", "torque", "air_flow",
        "fuel_flow", "air_fuel_ratio", "manifold_pressure", "ambient_inlet_temp"
    ]
    target_cols = [
        "brake_power", "brake_thermal_efficiency", "bsfc",
        "nox", "co", "hc", "smoke_opacity", "co2"
    ]

    X = df[feature_cols]
    y = df[target_cols]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, shuffle=True
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, shuffle=True
    )

    print(f"Train set:      {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set:       {X_test.shape[0]} samples")

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    # Transform Validation and Test sets
    X_val_scaled  = scaler_X.transform(X_val)
    y_val_scaled  = scaler_y.transform(y_val)
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    train_df = pd.DataFrame(
        np.hstack([X_train_scaled, y_train_scaled]),
        columns=[f"in_{c}" for c in feature_cols] + [f"out_{c}" for c in target_cols]
    )
    val_df = pd.DataFrame(
        np.hstack([X_val_scaled, y_val_scaled]),
        columns=[f"in_{c}" for c in feature_cols] + [f"out_{c}" for c in target_cols]
    )
    test_df = pd.DataFrame(
        np.hstack([X_test_scaled, y_test_scaled]),
        columns=[f"in_{c}" for c in feature_cols] + [f"out_{c}" for c in target_cols]
    )

    train_df.to_csv(os.path.join(output_dir, "train_scaled.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_scaled.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_scaled.csv"), index=False)

    X_test.to_csv(os.path.join(output_dir, "X_test_raw.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test_raw.csv"), index=False)

    joblib.dump(scaler_X, "models/scaler_X.pkl")
    joblib.dump(scaler_y, "models/scaler_y.pkl")

    print("\n--- Pipeline Completed Successfully ---")
    print(f"Processed CSVs saved in: {output_dir}/")
    print("Scalers exported to:     models/scaler_X.pkl & models/scaler_y.pkl")

if __name__ == "__main__":
    run_preprocessing()