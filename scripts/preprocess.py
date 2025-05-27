import pandas as pd
import yaml
import os

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# =============================================================================

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    # - keep the target and numeric features only
    target_col = "fraud_bool"
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if target_col not in numeric_cols:
        raise ValueError(f"'{target_col}' must be in numeric columns.")

    # - reorder columns to put the target first
    reordered_cols = [target_col] + [col for col in numeric_cols if col != target_col]
    df = df[reordered_cols]

    # - fill missing values if any (can be adjusted as needed)
    df.fillna(0, inplace=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    config = load_config()
    preprocess_data(config["data"]["raw"], config["data"]["processed"])
