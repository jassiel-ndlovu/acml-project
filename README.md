# Financial Fraud Detection using Autoencoder

This project detects fraudulent financial behavior using an autoencoder trained on normal transaction data from the **Bank Account Fraud Dataset Suite (NeurIPS 2022)**. Download [here](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022/data?select=Variant+IV.csv).

Note that in dir ```/data/splits/``` you can find ```fraud.csv``` and ```normal.csv``` data points based on the ```Base.csv``` variant of the dataset. Note the following structure of the ```acml-project``` directory:

```bash
acml-project/
├── data/
│   ├── raw/
│   │   └── Base.csv                    # Variants of the dataset
│   │   └── Variant I.csv
│   │   └── Variant II.csv
│   │   └── Variant III.csv
│   │   └── Variant IV.csv
│   │   └── Variant V.csv
│   ├── processed/
│   │   └── cleaned_base.csv            # Preprocessing clean
│   └── splits/
│       ├── test/                       # Test data set (25%)
│       │   └── test.csv
│       ├── train/                      # Train data set (50%)
│       │   └── fraud.csv               # Fraudulent data
│       │   └── normal.csv              # Normal (non-fraudulent) data
│       ├── validation/                 # Validation data set (25%)
│           └── validation.csv
├── outputs/
│   └── checkpoints/
│       └── autoencoder.pt              
├── scripts/
│   ├── preprocess.py                   
│   ├── split_data.py                   
│   ├── train_autoencoder.py           
│   └── test.py                         
├── models/
│   └── autoencoder.py                  
├── performance/
│   ├── train/
│   │   └── train_eval_metrics.txt  
│   │   └── training_loss.png
│   │   └── val_mse_distribution.png  
│   │   └── val_roc_curve.png 
│   ├── test/
│   │   └── confusion_matrix.png  
│   │   └── training_loss.png
│   │   └── metrics_report.txt  
│   │   └── mse_distribution.png
│   │   └── roc_curve.png
│   └── config.yaml                     
├── requirements.txt
├── .gitignore
├── config.yaml  
├── visualise.ipynb                                            
└── README.md                               
```

---

## How to Run the Project

Run the following scripts **in order**:

### 1. Preprocess the raw data
```bash
python scripts/preprocess.py
```

### 2. Split into normal and fraudulent data
```bash
python scripts/split_data.py
```

### 3. Train the autoencoder on normal data
```bash
python scripts/train_autoencoder.py
```

### 4. Evaluate
```bash
python scripts/test.py
```

---

## Setup

### Clone repo
```bash
git clone https://github.com/jassiel-ndlovu/acml-project.git
cd acml-project
```

### Install dependencies
```bash
pip install -r requirements.txt
```