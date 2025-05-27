import pandas as pd
import yaml
import os

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
# =============================================================================

def split_data(input_path, output_dir, normal_path, fraud_path):
    df = pd.read_csv(input_path)

    if "fraud_bool" not in df.columns:
        raise ValueError("'fraud_bool' column not found in the dataset.")

    normal = df[df["fraud_bool"] == 0].drop(columns=["fraud_bool"])
    fraud = df[df["fraud_bool"] == 1].drop(columns=["fraud_bool"])

    os.makedirs(output_dir, exist_ok=True)
    normal.to_csv(normal_path, index=False)
    fraud.to_csv(fraud_path, index=False)

    print(f"Saved {len(normal)} normal records to {normal_path}")
    print(f"Saved {len(fraud)} fraudulent records to {fraud_path}")

if __name__ == "__main__":
    config = load_config()
    split_data(
        input_path=config["data"]["processed"],
        output_dir=config["data"]["split_dir"],
        normal_path=config["data"]["normal_data"],
        fraud_path=config["data"]["fraud_data"]
    )
