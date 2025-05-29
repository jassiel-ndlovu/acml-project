import pandas as pd
import yaml
import os
from yaspin import yaspin

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# =============================================================================
# Read Preprocessing summary in the `visualise.ipnyb` file.
# =============================================================================

def preprocess_data(input_path, output_path):
    with yaspin(text="Reading raw data...", color="white") as spinner:
        df = pd.read_csv(input_path)
        spinner.ok("✔ ")

    # - (1) encoding maps for  non-numerical features
    category_maps = {
        "payment_type": {"AA": 1, "AB": 2, "AC": 3, "AD": 4, "AE": 5},
        "employment_status": {"CA": 1, "CB": 2, "CC": 3, "CD": 4, "CF": 5},
        "housing_status": {"BA": 1, "BB": 2, "BC": 3, "BD": 4, "BE": 5},
        "source": {"INTERNET": 1, "TELEAPP": 2},
        "device_os": {"linux": 1, "windows": 2, "macintosh": 3, "x11": 4, "other": 5}
    }

    with yaspin(text="Encoding categorical columns...", color="white") as spinner:
        for col, mapping in category_maps.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        spinner.ok("✔ ")

    # - (2) fill in missing values with 0's
    with yaspin(text="Filling missing values...", color="white") as spinner:
        df.fillna(0, inplace=True)
        spinner.ok("✔ ")

    with yaspin(text="Saving preprocessed data...", color="white") as spinner:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        spinner.ok("✔ ")
        print(f"Preprocessed data saved to {output_path}")

# =============================================================================

if __name__ == "__main__":
    config = load_config()
    preprocess_data(config["data"]["raw"], config["data"]["processed"])
