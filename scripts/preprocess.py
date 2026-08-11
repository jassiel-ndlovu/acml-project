"""Step 2 - Fit the Preprocessor on train-normal and transform every split.

Saves NumPy arrays consumed by training/evaluation, plus the fitted preprocessor.
"""
import os
import sys
import yaml
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import Preprocessor, TARGET


def main(cfg="config.yaml"):
    c = yaml.safe_load(open(cfg))
    sd = c["data"]["split_dir"]
    train = pd.read_csv(os.path.join(sd, "train.csv"))
    val = pd.read_csv(os.path.join(sd, "val.csv"))
    test = pd.read_csv(os.path.join(sd, "test.csv"))
    train_normal = train[train[TARGET] == 0].copy()

    pre = Preprocessor().fit(train, train_normal)
    print("Feature dim after encoding:", pre.input_dim)

    proc = c["data"]["processed_dir"]
    os.makedirs(proc, exist_ok=True)
    np.save(os.path.join(proc, "X_train.npy"), pre.transform(train_normal).values.astype(np.float32))
    np.save(os.path.join(proc, "X_val.npy"), pre.transform(val).values.astype(np.float32))
    np.save(os.path.join(proc, "y_val.npy"), val[TARGET].values)
    np.save(os.path.join(proc, "X_test.npy"), pre.transform(test).values.astype(np.float32))
    np.save(os.path.join(proc, "y_test.npy"), test[TARGET].values)
    pre.save(os.path.join(proc, "preprocessor.json"))
    print("Saved processed arrays and preprocessor to", proc)


if __name__ == "__main__":
    main()
