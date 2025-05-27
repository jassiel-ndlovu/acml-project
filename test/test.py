import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from model.autoencoder import Autoencoder

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def evaluate():
    config = load_config()
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # - load trained model
    model = torch.load(config["data"]["model_path"], map_location=device)
    model.eval()

    # - load datasets
    normal = pd.read_csv(config["data"]["normal_data"]).values.astype(np.float32)
    fraud = pd.read_csv(config["data"]["fraud_data"]).values.astype(np.float32)

    #  - combine for test
    X_test = np.vstack([normal, fraud])
    y_test = np.hstack([np.zeros(len(normal)), np.ones(len(fraud))])

    X_tensor = torch.tensor(X_test).to(device)
    with torch.no_grad():
        recon = model(X_tensor)
        mse = ((X_tensor - recon) ** 2).mean(dim=1).cpu().numpy()

    auc = roc_auc_score(y_test, mse)
    print(f"AUC-ROC Score for Fraud Detection: {auc:.4f}")

if __name__ == "__main__":
    evaluate()
