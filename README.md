# Financial Fraud Detection using Autoencoder

This project detects fraudulent financial behavior using an autoencoder trained on normal transaction data from the **Bank Account Fraud Dataset Suite (NeurIPS 2022)**.

Note that in dir ```/data/splits/``` you can find ```fraud.csv``` and ```normal.csv``` data points based on the ```Base.csv``` variant of the dataset.

---

## How to Run the Project

Run the following scripts **in order**:

### 1. Preprocess the raw data
```bash
python -m scripts.preprocess
```

### 2. Split into normal and fraudulent data
```bash
python -m scripts.split_data
```

### 3. Train the autoencoder on normal data
```bash
python -m scripts.train_autoencoder
```

### 4. Evaluate
```bash
python -m scripts.evaluate
```

---

## Setup

### Clone repo
```bash
git clone https://github.com/jassiel-ndlovu/acml-project.git
cd acml-project
```

### Download the dataset
```bash
curl -L -o data.zip https://www.kaggle.com/api/v1/datasets/download/sgpjesus/bank-account-fraud-dataset-neurips-2022
```
#### For Linux/macOS
```bash
unzip data.zip -d data/raw/
```

#### For Windows (PowerShell)
```powershell
Expand-Archive -Path data.zip -DestinationPath data/raw
```

### Install dependencies
```bash
pip install -r requirements.txt
```