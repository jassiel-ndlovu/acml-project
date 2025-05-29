import pandas as pd
import yaml
import os
from yaspin import yaspin
from sklearn.model_selection import train_test_split

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
# =============================================================================

def split_data(input_path, output_dir, normal_path, fraud_path, test_path, val_path):
    '''
    First we will split the data for training, validation and test. 
    We use the 50-25-25 split. That is, 50% of the dataset for training
    25% for validation and the other 25% for testing. 

    For training, we split the data into `normal` and `fraud` to train 
    the autoencoder on the normal dataset.
    '''
    df = pd.read_csv(input_path)

    # - 50-25-25 split
    with yaspin(text="Splitting into train, val and test datasets...", color="white") as spinner:
        train_df, temp_df = train_test_split(df, test_size=0.5, random_state=42, stratify=df["fraud_bool"])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["fraud_bool"])
        spinner.ok("✔ ")

    # - save validation and test datasets
    with yaspin(text="Saving datasets...", color="white") as spinner:
        os.makedirs(output_dir, exist_ok=True)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        print(f"Saved validation set to {val_path} ({len(val_df)} records)")
        print(f"Saved test set to {test_path} ({len(test_df)} records)")
        spinner.ok("✔ ")

    # - train data split (we drop the fraud_bool column)
    with yaspin(text="Splitting training data...", color="white") as spinner:
        normal = train_df[train_df["fraud_bool"] == 0].drop(columns=["fraud_bool"])
        fraud = train_df[train_df["fraud_bool"] == 1].drop(columns=["fraud_bool"])
        spinner.ok("✔ ")

    # - save the fraud and normal dataset
    with yaspin(text="Saving datasets...", color="white") as spinner:
        os.makedirs(output_dir, exist_ok=True)
        normal.to_csv(normal_path, index=False)
        fraud.to_csv(fraud_path, index=False)
        spinner.ok("✔ ")

    print(f"Saved normal records to {normal_path} ({len(normal)} records)")
    print(f"Saved fraudulent records to {fraud_path} ({len(fraud)} records)")

# =============================================================================

if __name__ == "__main__":
    config = load_config()
    split_data(
        input_path=config["data"]["processed"],
        output_dir=config["data"]["split_dir"],
        normal_path=config["data"]["normal_data"],
        fraud_path=config["data"]["fraud_data"],
        test_path=config["data"]["test_data"],
        val_path=config["data"]["validation_data"]
    )
