import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import model.autoencoder as Autoencoder

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# =============================================================================

from yaspin import yaspin

def evaluate():
    config = load_config()
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    with yaspin(text="Loading model and test data...", color="green") as spinner:
        model = torch.load(config["data"]["model_path"], map_location=device, weights_only=False)
        model.eval()

        df = pd.read_csv(config["data"]["test_data"])
        y_test = df["fraud_bool"].values
        X_test = df.drop(columns=["fraud_bool"]).values.astype(np.float32)
        X_tensor = torch.tensor(X_test).to(device)
        spinner.ok("✔")

    with yaspin(text="Running inference...", color="green") as spinner:
        with torch.no_grad():
            recon = model(X_tensor)
            mse = ((X_tensor - recon) ** 2).mean(dim=1).cpu().numpy()
        spinner.ok("✔")

    with yaspin(text="Computing metrics and saving visualizations...", color="green") as spinner:
        perf_dir = config["performance"]["test_path"]
        os.makedirs(perf_dir, exist_ok=True)

        fpr, tpr, thresholds = roc_curve(y_test, mse)
        auc_score = roc_auc_score(y_test, mse)
        best_thresh = thresholds[np.argmax(tpr - fpr)]

        # ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve"); plt.grid(True); plt.legend()
        plt.savefig(os.path.join(perf_dir, "roc_curve.png")); plt.close()

        # MSE distribution
        plt.figure()
        plt.hist(mse[y_test == 0], bins=50, alpha=0.6, label="Normal", color="green")
        plt.hist(mse[y_test == 1], bins=50, alpha=0.6, label="Fraud", color="red")
        plt.axvline(best_thresh, color="black", linestyle="--", label=f"Threshold: {best_thresh:.4f}")
        plt.xlabel("Reconstruction MSE"); plt.ylabel("Frequency"); plt.title("MSE Distribution"); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(perf_dir, "mse_distribution.png")); plt.close()

        # Classification metrics
        y_pred = (mse > best_thresh).astype(int)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"\nEvaluation Metrics:")
        print(f"AUC-ROC Score: {auc_score:.4f}")
        print(f"Threshold: {best_thresh:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fraud"])
        disp.plot(cmap="Blues"); plt.title("Confusion Matrix")
        plt.savefig(os.path.join(perf_dir, "confusion_matrix.png")); plt.close()

        with open(os.path.join(perf_dir, "metrics_report.txt"), "w") as f:
            f.write(f"AUC-ROC Score: {auc_score:.4f}\n")
            f.write(f"Threshold: {best_thresh:.4f}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Precision: {prec:.4f}\n")
            f.write(f"Recall: {rec:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))
        spinner.ok("✔")

# =============================================================================

if __name__ == "__main__":
    evaluate()
