"""Step 3 - Train the autoencoder on normal data with early stopping (CPU or GPU)."""
import os
import sys
import time
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.autoencoder import Autoencoder


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main(cfg="config.yaml"):
    c = yaml.safe_load(open(cfg))
    set_seed(c["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", dev)

    proc = c["data"]["processed_dir"]
    X = np.load(os.path.join(proc, "X_train.npy"))
    tr, es = train_test_split(X, test_size=0.1, random_state=c["seed"])
    loader = DataLoader(TensorDataset(torch.tensor(tr)),
                        batch_size=c["training"]["batch_size"], shuffle=True)
    es_t = torch.tensor(es).to(dev)

    model = Autoencoder(X.shape[1], dropout=c["model"]["dropout"]).to(dev)
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=c["training"]["learning_rate"],
                           weight_decay=c["training"]["weight_decay"])

    best, bad, patience = np.inf, 0, c["training"]["patience"]
    os.makedirs(os.path.dirname(c["data"]["model_path"]), exist_ok=True)
    hist = []
    t0 = time.time()
    for ep in range(c["training"]["num_epochs"]):
        model.train(); tot = 0.0
        for (xb,) in loader:
            xb = xb.to(dev); opt.zero_grad()
            loss = crit(model(xb), xb); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            vloss = crit(model(es_t), es_t).item()
        hist.append((ep + 1, tot / len(tr), vloss))
        if vloss < best - 1e-6:
            best, bad = vloss, 0
            torch.save(model.state_dict(), c["data"]["model_path"])
        else:
            bad += 1
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"epoch {ep+1:02d}  train {tot/len(tr):.5f}  es {vloss:.5f}  (best {best:.5f})")
        if bad >= patience:
            print("early stop @", ep + 1); break
    print("train time: %.1fs on %s" % (time.time() - t0, dev))
    os.makedirs(c["performance"]["train_path"], exist_ok=True)
    np.save(os.path.join(c["performance"]["train_path"], "loss_history.npy"), np.array(hist))
    print("Saved model to", c["data"]["model_path"])


if __name__ == "__main__":
    main()
