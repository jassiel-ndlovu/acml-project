import sys
import os
# Add project root to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model.autoencoder import Autoencoder
import os
import random

# config file
def load_config(path="config.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)
    
# =============================================================================

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train():
    config = load_config()
    set_seed(config["seed"])

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(config["data"]["normal_data"]).values.astype(np.float32)
    dataset = TensorDataset(torch.tensor(df))
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)

    input_dim = config["model"]["input_dim"]
    model = Autoencoder(input_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    model.train()
    for epoch in range(config["training"]["num_epochs"]):
        epoch_loss = 0
        for (batch,) in loader:
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(loader):.4f}")

    # Save latest checkpoint and final model
    os.makedirs(os.path.dirname(config["data"]["checkpoint_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(config["data"]["model_path"]), exist_ok=True)

    torch.save(model.state_dict(), config["data"]["checkpoint_path"])
    torch.save(model, config["data"]["model_path"])
    print(f"Model saved to {config['data']['model_path']}")

if __name__ == "__main__":
    train()
