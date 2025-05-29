import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def plot_test():
    config = load_config()
    perf_dir = config["performance"]["test_path"]
    os.makedirs(perf_dir, exist_ok=True)

    # - load data with columns: mse, label
    df = pd.read_csv(os.path.join(perf_dir, "test_metrics.csv")) 
    mse = df["mse"].values
    y_true = df["label"].values

    #  -threshold from config
    best_thresh = config["performance"]["threshold"]

    # - compute ROC & AUC
    fpr, tpr, thresholds = roc_curve(y_true, mse)
    auc_score = roc_auc_score(y_true, mse)

    # - ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Test ROC Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(perf_dir, "roc_curve.png"))
    plt.close()

    # - MSE distribution
    plt.figure()
    plt.hist(mse[y_true == 0], bins=50, alpha=0.6, label="Normal", color="green")
    plt.hist(mse[y_true == 1], bins=50, alpha=0.6, label="Fraud", color="red")
    plt.axvline(best_thresh, color="black", linestyle="--", label=f"Threshold: {best_thresh:.4f}")
    plt.xlabel("Reconstruction MSE")
    plt.ylabel("Frequency")
    plt.title("Test MSE Distribution")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(perf_dir, "mse_distribution.png"))
    plt.close()

    # - confusion Matrix
    y_pred = (mse > best_thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fraud"])
    disp.plot(cmap="Blues")
    plt.title("Test Confusion Matrix")
    plt.savefig(os.path.join(perf_dir, "confusion_matrix.png"))
    plt.close()

if __name__ == "__main__":
    plot_test()
