# AURA — Unsupervised Autoencoder Fraud Detection

Detecting fraudulent bank-account-opening applications **without fraud labels**, using a
deep autoencoder trained only on legitimate transactions. Anomalies are flagged by high
reconstruction error. Built on the **Bank Account Fraud (BAF, NeurIPS 2022)** benchmark
(1,000,000 records, 1.10% fraud).

> This is a corrected, reproducible rebuild of an earlier course project. Every number in
> the report and notebook comes from a real training run on the full dataset — nothing is
> fabricated.

## Headline results (held-out test set, 250k records)

| Model | ROC-AUC | PR-AUC (AP) | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **Autoencoder** | **0.592** | **0.016** | 0.022 | 0.153 | **0.038** |
| Isolation Forest (baseline) | 0.553 | 0.014 | 0.016 | 0.135 | 0.029 |
| Random | 0.500 | 0.011 | 0.011 | — | — |

The autoencoder beats a standard Isolation Forest baseline and the random floor on every
metric. **Average precision (PR-AUC) is the honest headline metric** for a 1.1%-prevalence
problem — a "never fraud" classifier already scores 98.9% accuracy, so accuracy is
reported only to show why it is misleading. Absolute separation is modest because the
model is fully unsupervised on a deliberately hard, biased benchmark (published
*supervised* baselines reach ROC-AUC ≈ 0.85); AURA's role is a **label-free first-line
filter** that narrows the review population.

*Exact metrics vary by ±0.01 across hardware / PyTorch versions due to nondeterminism; a
representative run gives ROC-AUC ≈ 0.58–0.59, PR-AUC ≈ 0.016.*

## What was fixed vs. the original project

- **Real preprocessing.** The original `preprocess.py` only encoded categoricals and did
  `fillna(0)`. This version adds Min-Max scaling, 1st/99th-percentile outlier clipping,
  and constant-column removal — the steps the original *report* described but the code
  never performed (which is why the original threshold was ~28 and ROC-AUC 0.44, *worse
  than random*).
- **`-1` missing-value sentinels.** BAF encodes missing as `-1` (71% of
  `prev_address_months_count`). We add missing-indicator features + median imputation
  instead of treating `-1` as a real magnitude.
- **Correct categorical cardinalities.** `employment_status` and `housing_status` each
  have **7** categories (CA–CG, BA–BG); the original mapped only 5, silently corrupting
  rows to NaN→0. Now one-hot encoded from data.
- **No data leakage.** Split happens **before** any transform is fit; all statistics are
  fit on the training-normal subset only.
- **Honest thresholding + evaluation.** Threshold chosen on **validation** (not test);
  PR-AUC reported alongside ROC-AUC; Isolation Forest baseline for context.
- **Bug fixes.** The old `test.py` loaded a full model but training saved a `state_dict`
  (crash); `generate_test_plots.py` read a non-existent config key. Both fixed.

## Repository layout

```
├── AURA_fraud_detection.ipynb   # ★ end-to-end Colab notebook (GPU-ready), with outputs
├── AURA_report.pdf              # IEEE-style report
├── config.yaml                  # all paths & hyperparameters
├── model/autoencoder.py         # the autoencoder
├── src/preprocessing.py         # leakage-free Preprocessor (fit/transform/save/load)
├── scripts/
│   ├── split_data.py            # 1. stratified 50/25/25 split
│   ├── preprocess.py            # 2. fit on train-normal, transform all splits
│   ├── train_autoencoder.py     # 3. train w/ early stopping (CPU or CUDA)
│   └── evaluate.py              # 4. threshold on val, test metrics + baseline + plots
├── figures/                     # generated figures
└── outputs/autoencoder_final.pt # trained weights
```

## Quickstart

### Option A — Google Colab (recommended, free GPU)
Open `AURA_fraud_detection.ipynb` in Colab, set `Runtime → Change runtime type → T4 GPU`,
and run all cells. The notebook auto-detects CUDA and guides you through getting the data.

### Option B — Local / any CUDA machine
```bash
pip install -r requirements.txt

# Download Base.csv from Kaggle into data/raw/ :
# https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022

python scripts/split_data.py         # 1. split
python scripts/preprocess.py         # 2. preprocess (fit on train-normal)
python scripts/train_autoencoder.py  # 3. train  (uses GPU automatically if available)
python scripts/evaluate.py           # 4. evaluate + baseline + figures
```
Training takes < 3 min on CPU and seconds on a GPU (the model is ~13k parameters).

## Method in one paragraph
Split 50/25/25 (stratified). Fit preprocessing on the *normal* training rows only. Train a
symmetric autoencoder (57→64→32→16→32→64→57, BatchNorm + ReLU + Dropout, sigmoid output)
to minimise MSE reconstruction of normal transactions, with early stopping on held-out
normal loss. Score each transaction by its reconstruction error; flag those above a
validation-tuned threshold. Compare against Isolation Forest and the random baseline using
ROC-AUC and PR-AUC.

## Author
**Nkosenhle Ndlovu** (2539199) — University of the Witwatersrand.
Dataset: Jesus et al., *"Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets
for ML Evaluation,"* NeurIPS 2022.
