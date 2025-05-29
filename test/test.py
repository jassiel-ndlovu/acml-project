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
        best_thresh = config["model"]["threshold"]

        y_pred = (mse > best_thresh).astype(int)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # - summary
        print(f"\nEvaluation Metrics:")
        print(f"AUC-ROC Score: {auc_score:.4f}")
        print(f"Threshold: {best_thresh:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # - metrics
        with open(os.path.join(perf_dir, "metrics_report.txt"), "w") as f:
            f.write(f"AUC-ROC Score: {auc_score:.4f}\n")
            f.write(f"Threshold: {best_thresh:.4f}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Precision: {prec:.4f}\n")
            f.write(f"Recall: {rec:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

        # - raw data for graphing
        pd.DataFrame({
            "reconstruction_mse": mse,
            "true_label": y_test,
            "predicted_label": y_pred
        }).to_csv(os.path.join(perf_dir, "raw_predictions.csv"), index=False)

        pd.DataFrame({
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds
        }).to_csv(os.path.join(perf_dir, "roc_data.csv"), index=False)

        np.savetxt(os.path.join(perf_dir, "confusion_matrix.txt"), cm, fmt='%d')

        pd.DataFrame({"mse": mse, "label": y_test}).to_csv(
        os.path.join(perf_dir, "test_metrics.csv"), index=False)

        spinner.ok("✔")


# =============================================================================

if __name__ == "__main__":
    evaluate()
