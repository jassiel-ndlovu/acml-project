"""Step 4 - Threshold on validation, evaluate on test, fit baseline, save metrics+plots."""
import os
import sys
import json
import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                             precision_recall_curve, confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score, accuracy_score)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.autoencoder import Autoencoder

BLUE, VERM, ORANGE, GREY = "#0072B2", "#D55E00", "#E69F00", "#666666"


def recon_mse(model, X, dev, bs=8192):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.tensor(X[i:i + bs]).to(dev)
            out.append(((xb - model(xb)) ** 2).mean(1).cpu().numpy())
    return np.concatenate(out)


def evaluate(y, s, thr):
    yp = (s > thr).astype(int)
    return dict(roc_auc=float(roc_auc_score(y, s)), pr_auc=float(average_precision_score(y, s)),
                threshold=float(thr), accuracy=float(accuracy_score(y, yp)),
                precision=float(precision_score(y, yp, zero_division=0)),
                recall=float(recall_score(y, yp, zero_division=0)),
                f1=float(f1_score(y, yp, zero_division=0)),
                confusion=confusion_matrix(y, yp).tolist(),
                report=classification_report(y, yp, target_names=["Normal", "Fraud"], zero_division=0))


def main(cfg="config.yaml"):
    c = yaml.safe_load(open(cfg))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc, fig, perf = c["data"]["processed_dir"], c["performance"]["fig_path"], c["performance"]["test_path"]
    os.makedirs(fig, exist_ok=True); os.makedirs(perf, exist_ok=True)

    X_train = np.load(os.path.join(proc, "X_train.npy"))
    X_val = np.load(os.path.join(proc, "X_val.npy")); y_val = np.load(os.path.join(proc, "y_val.npy"))
    X_test = np.load(os.path.join(proc, "X_test.npy")); y_test = np.load(os.path.join(proc, "y_test.npy"))

    model = Autoencoder(X_train.shape[1], dropout=c["model"]["dropout"]).to(dev)
    model.load_state_dict(torch.load(c["data"]["model_path"], map_location=dev))

    val_mse = recon_mse(model, X_val, dev)
    test_mse = recon_mse(model, X_test, dev)

    p, r, thr = precision_recall_curve(y_val, val_mse)
    best_thr = float(thr[np.argmax((2 * p * r / (p + r + 1e-12))[:-1])])
    ae = evaluate(y_test, test_mse, best_thr)

    # persist threshold back to config
    c["model"]["threshold"] = best_thr
    yaml.safe_dump(c, open(cfg, "w"), sort_keys=False)

    # Isolation Forest baseline
    iso = IsolationForest(n_estimators=100, contamination="auto", random_state=c["seed"], n_jobs=-1)
    iso.fit(X_train)
    iso_test = -iso.score_samples(X_test)
    p2, r2, t2 = precision_recall_curve(y_val, -iso.score_samples(X_val))
    iso_thr = float(t2[np.argmax((2 * p2 * r2 / (p2 + r2 + 1e-12))[:-1])])
    iso_res = evaluate(y_test, iso_test, iso_thr)

    results = dict(autoencoder=dict(test=ae), isolation_forest=dict(test=iso_res), device=str(dev))
    json.dump(results, open(os.path.join(perf, "results.json"), "w"), indent=2)

    # ---- plots ----
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    base = float(y_test.mean())
    f, t, _ = roc_curve(y_test, test_mse); fi, ti, _ = roc_curve(y_test, iso_test)
    plt.figure(figsize=(5, 4.6))
    plt.plot(f, t, color=BLUE, lw=2, label=f"Autoencoder (AUC={ae['roc_auc']:.3f})")
    plt.plot(fi, ti, color=ORANGE, lw=2, label=f"Isolation Forest (AUC={iso_res['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], color=GREY, ls=":", label="Random (0.500)")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve (Test)")
    plt.legend(frameon=False, loc="lower right"); plt.grid(alpha=.25)
    plt.savefig(os.path.join(fig, "roc_curve.png"), bbox_inches="tight"); plt.close()

    pr, rc, _ = precision_recall_curve(y_test, test_mse); pi, ri, _ = precision_recall_curve(y_test, iso_test)
    plt.figure(figsize=(5, 4.6))
    plt.plot(rc, pr, color=BLUE, lw=2, label=f"Autoencoder (AP={ae['pr_auc']:.3f})")
    plt.plot(ri, pi, color=ORANGE, lw=2, label=f"Isolation Forest (AP={iso_res['pr_auc']:.3f})")
    plt.axhline(base, color=GREY, ls=":", label=f"Random ({base:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve (Test)")
    plt.legend(frameon=False); plt.grid(alpha=.25)
    plt.savefig(os.path.join(fig, "pr_curve.png"), bbox_inches="tight"); plt.close()

    print("\n===== AUTOENCODER (TEST) =====")
    for k in ["roc_auc", "pr_auc", "threshold", "accuracy", "precision", "recall", "f1"]:
        print(f"  {k:10s}: {ae[k]:.4f}")
    print(ae["report"])
    print("Isolation Forest: ROC-AUC %.4f  PR-AUC %.4f" % (iso_res["roc_auc"], iso_res["pr_auc"]))
    print("Saved metrics to", perf, "and figures to", fig)


if __name__ == "__main__":
    main()
