"""Step 1 - Split raw Base.csv into stratified 50/25/25 train/val/test.

Splitting happens BEFORE any preprocessing so that scaling/imputation statistics
are never fit on validation or test data (no leakage).
"""
import os
import sys
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import TARGET


def main(cfg="config.yaml"):
    c = yaml.safe_load(open(cfg))
    seed = c["seed"]
    df = pd.read_csv(c["data"]["raw"])
    print(f"Loaded {df.shape}, fraud rate {100 * df[TARGET].mean():.4f}%")

    train_df, temp = train_test_split(df, test_size=0.5, random_state=seed, stratify=df[TARGET])
    val_df, test_df = train_test_split(temp, test_size=0.5, random_state=seed, stratify=temp[TARGET])

    out = c["data"]["split_dir"]
    os.makedirs(out, exist_ok=True)
    train_df.to_csv(os.path.join(out, "train.csv"), index=False)
    val_df.to_csv(os.path.join(out, "val.csv"), index=False)
    test_df.to_csv(os.path.join(out, "test.csv"), index=False)
    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print(f"train-normal={int((train_df[TARGET] == 0).sum())}")


if __name__ == "__main__":
    main()
