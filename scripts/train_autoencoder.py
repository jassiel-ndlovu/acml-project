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

    loss_file = os.path.join(config["performance"]["train_path"], "losses.txt")
    os.makedirs(config["performance"]["train_path"], exist_ok=True)

    # - training
    with open(loss_file, "w") as lf:
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

                # - CSV format: epoch,loss
                lf.write(f"{epoch+1},{avg_loss:.6f}\n")  
                spinner.write(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
            spinner.ok("✔ ")

    # - model weights
    with yaspin(text="Saving trained model...", color="green") as spinner:
        os.makedirs(os.path.dirname(config["data"]["model_path"]), exist_ok=True)
        torch.save(model.state_dict(), config["data"]["model_path"])
        spinner.ok("✔ ")
        print(f"Model saved to {config['data']['model_path']}")

    # - evaluation and metrics
    with yaspin(text="Evaluating on validation set...", color="green") as spinner:
        model.eval()
        with torch.no_grad():
            val_tensor = torch.tensor(val_X).to(device)
            recon = model(val_tensor)
            mse = ((val_tensor - recon) ** 2).mean(dim=1).cpu().numpy()
        fpr, tpr, thresholds = roc_curve(val_y, mse)
        auc_score = roc_auc_score(val_y, mse)

        # - new threshold, overwrite `config.yaml`
        best_thresh = thresholds[np.argmax(tpr - fpr)]
        config["model"]["threshold"] = float(best_thresh)

        with open("config.yaml", "w") as f:
            yaml.safe_dump(config, f)

        print(f"Saved best threshold to config.yaml: {best_thresh:.4f}")

        spinner.ok("✔ ")

    # - metrics (summary)
    with yaspin(text="Saving performance metrics...", color="green") as spinner:
        metrics_file = os.path.join(config["performance"]["train_path"], "metrics.txt")
        with open(metrics_file, "w") as f:
            f.write(f"AUC_ROC={auc_score:.6f}\n")
            f.write(f"Best_Threshold={best_thresh:.6f}\n")
            f.write(f"Final_Training_Loss={loss_history[-1]:.6f}\n")
        spinner.ok("✔ ")
    
    # - MSEs of training data for later plotting
    with yaspin(text="Computing and saving training MSEs...", color="green") as spinner:
        # - re-load training data
        train_data_df = pd.read_csv(config["data"]["normal_data"])
        X_train = train_data_df.values.astype(np.float32)

        # - all labels are supposed to be zero anyway
        y_train = np.zeros(X_train.shape[0])

        X_tensor = torch.tensor(X_train).to(device)
        model.eval()
        with torch.no_grad():
            recon = model(X_tensor)
            mse = ((X_tensor - recon) ** 2).mean(dim=1).cpu().numpy()

        # - save MSE and labels (all 0s) to CSV for later plotting
        perf_dir = config["performance"]["train_path"]
        os.makedirs(perf_dir, exist_ok=True)
        pd.DataFrame({"mse": mse, "label": y_train}).to_csv(
            os.path.join(perf_dir, "train_metrics.csv"), index=False
        )
        spinner.ok("✔")

# =============================================================================

if __name__ == "__main__":
    train()
