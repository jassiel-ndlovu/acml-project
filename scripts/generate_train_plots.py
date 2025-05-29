import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def plot_train():
    config = load_config()
    perf_dir = config["performance"]["train_path"]
    os.makedirs(perf_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(os.path.join(perf_dir, "train_metrics.csv"))  # assumes columns: mse, label
    mse = df["mse"].values
    y_true = df["label"].values

    # Compute ROC & AUC
    fpr, tpr, thresholds = roc_curve(y_true, mse)
    auc_score = roc_auc_score(y_true, mse)

    # Plot ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Training ROC Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(perf_dir, "roc_curve.png"))
    plt.close()

    # Plot MSE distribution
    plt.figure()
    plt.hist(mse[y_true == 0], bins=50, alpha=0.6, label="Normal", color="green")
    plt.hist(mse[y_true == 1], bins=50, alpha=0.6, label="Fraud", color="red")
    plt.xlabel("Reconstruction MSE")
    plt.ylabel("Frequency")
    plt.title("Training MSE Distribution")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(perf_dir, "mse_distribution.png"))
    plt.close()

if __name__ == "__main__":
    plot_train()
