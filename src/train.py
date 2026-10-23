import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import EngineANN

def load_data():
    train_df = pd.read_csv("data/processed/train_scaled.csv")
    val_df = pd.read_csv("data/processed/val_scaled.csv")
    test_df = pd.read_csv("data/processed/test_scaled.csv")

    X_train = torch.tensor(train_df.iloc[:, :7].values, dtype=torch.float32)
    y_train = torch.tensor(train_df.iloc[:, 7:].values, dtype=torch.float32)

    X_val = torch.tensor(val_df.iloc[:, :7].values, dtype=torch.float32)
    y_val = torch.tensor(val_df.iloc[:, 7:].values, dtype=torch.float32)

    X_test = torch.tensor(test_df.iloc[:, :7].values, dtype=torch.float32)
    y_test = torch.tensor(test_df.iloc[:, 7:].values, dtype=torch.float32)

    return X_train, y_train, X_val, y_val, X_test, y_test

def train_model():
    torch.manual_seed(42)
    np.random.seed(42)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = EngineANN(input_dim=7, output_dim=8)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

    epochs = 400
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    os.makedirs("models", exist_ok=True)
    best_model_path = "models/best_ann_model.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        epoch_train_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            epoch_val_loss = criterion(val_preds, y_val).item()
            val_losses.append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{epochs}] | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

    print(f"\nBest Validation Loss: {best_val_loss:.6f}")
    print(f"Model saved to: {best_model_path}")

    os.makedirs("reports/figures", exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, epochs + 1), train_losses, label="Training Loss", color="royalblue")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss", color="darkorange")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("ANN Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plot_path = "reports/figures/week3_loss_curve.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved loss curve to: {plot_path}")

if __name__ == "__main__":
    train_model()   