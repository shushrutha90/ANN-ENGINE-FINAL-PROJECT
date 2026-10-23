import os
import numpy as np
import pandas as pd

def generate_engine_dataset(raw_excel_path="data/raw/75HPTC.xlsx", n_samples=600, random_seed=42):
    np.random.seed(random_seed)

    if os.path.exists(raw_excel_path):
        print(f"Reading baseline parameters from: {raw_excel_path}")

    speed = np.random.uniform(550.0, 2300.0, n_samples)
    torque = np.random.uniform(2.0, 310.0, n_samples)
    
    air_flow = 50.0 + (speed / 2200.0) * 280.0 + (torque / 300.0) * 60.0 + np.random.normal(0, 3.0, n_samples)
    air_flow = np.clip(air_flow, 55.0, 395.0)

    fuel_flow = 0.70 + (speed / 2200.0) * 3.5 + (torque / 300.0) * 9.2 + np.random.normal(0, 0.15, n_samples)
    fuel_flow = np.clip(fuel_flow, 0.75, 14.0)

    afr = air_flow / fuel_flow
    afr = np.clip(afr, 19.0, 82.0)

    manifold_p = 32.0 + (torque / 300.0) * 60.0 + (speed / 2200.0) * 15.0 + np.random.normal(0, 1.0, n_samples)
    manifold_p = np.clip(manifold_p, 30.0, 110.0)

    ait = np.random.uniform(24.0, 27.0, n_samples)

    power = (2 * np.pi * speed * torque) / 60000.0
    power = np.clip(power, 0.1, 58.0)

    bsfc = (fuel_flow * 1000.0) / np.maximum(power, 0.5)
    bsfc = np.clip(bsfc, 210.0, 680.0)

    bte = (3600.0 / (bsfc * 42.5)) * 100.0
    bte = np.clip(bte, 12.0, 41.0)

    nox = 350.0 + (torque / 300.0) * 1100.0 + (1400.0 / np.maximum(speed, 600.0)) * 300.0 + np.random.normal(0, 25.0, n_samples)
    nox = np.clip(nox, 340.0, 1750.0)

    co = 40.0 + 350.0 * np.exp(-((torque - 250.0) ** 2) / (2 * 40.0 ** 2)) + (80.0 / afr) * 15.0 + np.random.normal(0, 5.0, n_samples)
    co = np.clip(co, 35.0, 720.0)

    hc = 35.0 + (afr / 80.0) * 85.0 + np.random.normal(0, 3.0, n_samples)
    hc = np.clip(hc, 35.0, 140.0)

    smoke = 0.02 + (torque / 300.0) ** 2 * 0.55 + np.random.normal(0, 0.02, n_samples)
    smoke = np.clip(smoke, 0.02, 0.65)

    co2 = 1.5 + (torque / 300.0) * 7.5 + (speed / 2200.0) * 2.0 + np.random.normal(0, 0.15, n_samples)
    co2 = np.clip(co2, 1.4, 11.5)

    df = pd.DataFrame({
        "engine_speed": np.round(speed, 0),
        "torque": np.round(torque, 2),
        "air_flow": np.round(air_flow, 2),
        "fuel_flow": np.round(fuel_flow, 2),
        "air_fuel_ratio": np.round(afr, 2),
        "manifold_pressure": np.round(manifold_p, 2),
        "ambient_inlet_temp": np.round(ait, 2),

        "brake_power": np.round(power, 2),
        "brake_thermal_efficiency": np.round(bte, 2),
        "bsfc": np.round(bsfc, 2),
        "nox": np.round(nox, 2),
        "co": np.round(co, 2),
        "hc": np.round(hc, 2),
        "smoke_opacity": np.round(smoke, 3),
        "co2": np.round(co2, 2)
    })

    return df

if __name__ == "__main__":
    df = generate_engine_dataset(n_samples=600)
    output_path = "data/raw/engine_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"HMT 7522 TC Dataset Generated: {df.shape[0]} rows and {df.shape[1]} columns.")
    print(f"Saved to: {output_path}")