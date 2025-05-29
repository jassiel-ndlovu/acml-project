import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from yaspin import yaspin
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, roc_curve
from model.autoencoder import Autoencoder

# =============================================================================
def load_config(path="config.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# =============================================================================

def train():
    config = load_config()
    set_seed(config["seed"])

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # - load data
    with yaspin(text="Loading training and validation data...", color="green") as spinner:
        train_data = pd.read_csv(config["data"]["normal_data"]).values.astype(np.float32)
        val_df = pd.read_csv(config["data"]["validation_data"])
        val_X = val_df.drop(columns=["fraud_bool"]).values.astype(np.float32)
        val_y = val_df["fraud_bool"].values
        spinner.ok("✔ ")

    # - call `Autoencoder`, setup model, Adam and dataloader
    train_loader = DataLoader(TensorDataset(torch.tensor(train_data)), batch_size=config["training"]["batch_size"], shuffle=True)
    input_dim = config["model"]["input_dim"]
    model = Autoencoder(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    loss_history = []

    with yaspin(text="Training autoencoder...", color="green") as spinner:
        model.train()
        for epoch in range(config["training"]["num_epochs"]):
            epoch_loss = 0
            for (batch,) in train_loader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_loss)
            spinner.write(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        spinner.ok("✔ ")

    # - save the model
    with yaspin(text="Saving trained model...", color="green") as spinner:
        os.makedirs(os.path.dirname(config["data"]["model_path"]), exist_ok=True)
        torch.save(model, config["data"]["model_path"])
        spinner.ok("✔ ")
        print(f"Model saved to {config['data']['model_path']}")

    # - evaluate on validation set (for hyperparameter tuning)
    with yaspin(text="Evaluating on validation set...", color="green") as spinner:
        model.eval()
        with torch.no_grad():
            val_tensor = torch.tensor(val_X).to(device)
            recon = model(val_tensor)
            mse = ((val_tensor - recon) ** 2).mean(dim=1).cpu().numpy()
        fpr, tpr, thresholds = roc_curve(val_y, mse)
        auc_score = roc_auc_score(val_y, mse)
        best_thresh = thresholds[np.argmax(tpr - fpr)]
        spinner.ok("✔ ")

    # - save the performance metrics and plots
    with yaspin(text="Saving plots and performance metrics...", color="green") as spinner:
        perf_dir = config["performance"]["train_path"]
        os.makedirs(perf_dir, exist_ok=True)

        # - MSE loss plot (accuracy)
        plt.figure()
        plt.plot(loss_history, label="Training MSE Loss", color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training MSE Loss")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(perf_dir, "training_loss.png"))
        plt.close()

        # - MSE distribution
        plt.figure()
        plt.hist(mse[val_y == 0], bins=50, alpha=0.6, label="Normal", color="green")
        plt.hist(mse[val_y == 1], bins=50, alpha=0.6, label="Fraud", color="red")
        plt.axvline(best_thresh, color="black", linestyle="--", label=f"Threshold: {best_thresh:.4f}")
        plt.xlabel("Reconstruction MSE")
        plt.ylabel("Frequency")
        plt.title("Validation MSE Distribution")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(perf_dir, "val_mse_distribution.png"))
        plt.close()

        # - ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Validation ROC Curve")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(perf_dir, "val_roc_curve.png"))
        plt.close()

        # - draw up comprehensive summary
        with open(os.path.join(perf_dir, "train_eval_metrics.txt"), "w") as f:
            f.write(f"AUC-ROC Score: {auc_score:.4f}\n")
            f.write(f"Best Threshold: {best_thresh:.4f}\n")
            f.write(f"Final Training Loss: {loss_history[-1]:.6f}\n")

        spinner.ok("✔ ")

# =============================================================================

if __name__ == "__main__":
    train()
