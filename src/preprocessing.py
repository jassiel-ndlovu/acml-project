"""Leakage-free preprocessing for the Bank Account Fraud (BAF) dataset.

The Preprocessor is *fit on the training-normal subset only* and then applied
unchanged to validation/test. It handles:
  - dropping the constant `device_fraud_count` column
  - the BAF `-1` missing-value sentinel (adds missing-indicator features + median impute)
  - 1st/99th-percentile outlier clipping on continuous features
  - Min-Max scaling of continuous features to [0, 1]
  - one-hot encoding of the 5 nominal categoricals (correct 5/7/7/2/5 cardinalities)
"""
import json
import numpy as np
import pandas as pd

TARGET = "fraud_bool"
CAT = ["payment_type", "employment_status", "housing_status", "source", "device_os"]
DROP_CONST = ["device_fraud_count"]
SENTINEL = ["prev_address_months_count", "current_address_months_count",
            "credit_risk_score", "bank_months_count",
            "session_length_in_minutes", "device_distinct_emails_8w"]


class Preprocessor:
    def __init__(self):
        self.cont, self.binary, self.cat_levels = [], [], {}
        self.median, self.clip_lo, self.clip_hi = {}, {}, {}
        self.smin, self.smax = {}, {}
        self.features = []

    def fit(self, train_df: pd.DataFrame, train_normal: pd.DataFrame):
        df = train_df.drop(columns=DROP_CONST, errors="ignore")
        num_cols = [c for c in df.columns if c not in CAT + [TARGET]]
        self.binary = [c for c in num_cols if set(df[c].dropna().unique()) <= {0, 1}]
        self.cont = [c for c in num_cols if c not in self.binary]
        self.cat_levels = {c: sorted(df[c].unique().tolist()) for c in CAT}

        f = train_normal.copy()
        for c in SENTINEL:
            if c in f:
                f[c] = f[c].replace(-1, np.nan)
        for c in self.cont:
            med = float(f[c].median())
            self.median[c] = med
            col = f[c].fillna(med)
            lo, hi = np.percentile(col, 1), np.percentile(col, 99)
            self.clip_lo[c], self.clip_hi[c] = float(lo), float(hi)
            col = col.clip(lo, hi)
            self.smin[c], self.smax[c] = float(col.min()), float(col.max())
        # establish canonical feature order
        self.features = self.transform(train_normal.head(1)).columns.tolist()
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        f = frame.drop(columns=DROP_CONST, errors="ignore").copy()
        for c in SENTINEL:
            if c in f:
                f[c + "_missing"] = (f[c] == -1).astype(float)
                f[c] = f[c].replace(-1, np.nan)
        for c in self.cont:
            col = f[c].fillna(self.median[c]).clip(self.clip_lo[c], self.clip_hi[c])
            rng = self.smax[c] - self.smin[c]
            f[c] = (col - self.smin[c]) / rng if rng > 0 else 0.0
        parts = [f[self.cont + self.binary + [s + "_missing" for s in SENTINEL]]]
        for c in CAT:
            d = pd.get_dummies(f[c], prefix=c)
            for lv in self.cat_levels[c]:
                if f"{c}_{lv}" not in d.columns:
                    d[f"{c}_{lv}"] = 0
            parts.append(d[[f"{c}_{lv}" for lv in self.cat_levels[c]]].astype(float))
        out = pd.concat(parts, axis=1)
        if self.features:
            out = out[self.features]
        return out

    @property
    def input_dim(self):
        return len(self.features)

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.__dict__, fh)

    @classmethod
    def load(cls, path):
        obj = cls()
        with open(path) as fh:
            obj.__dict__.update(json.load(fh))
        return obj
