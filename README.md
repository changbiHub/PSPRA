# PSPRA - Paralytic Shellfish Poisoning Risk Assessment

This is the official repository for the paper "Paralytic Shellfish Poisoning Risk Assessment in the West Coast of Canada". This repository contains the model implementation and training code used in the research.

## Important Notice

Due to data sharing agreements with the Canadian Food Inspection Agency (CFIA), we cannot provide the original dataset used in the paper. For demonstration purposes, we have created synthetic data based on the real data characteristics. **Please note that the synthetic data may not completely reflect the properties of the original data and may show different results from those reported in the paper.**

## Project Structure

```
PSPRA/
├── README.md
├── script/
│   ├── train.py          # Main training script
│   ├── nnModel.py        # Neural network model implementations
│   └── preprocessor.py   # Data preprocessing utilities
├── data/                 # Synthetic datasets
├── results/              # Training results and predictions
└── models/               # Saved trained models
```

## Environment Setup

You can set up the Python environment using either conda (recommended) or venv. Choose one of the methods below:

### Method 1: Using Conda (Recommended)

```bash
# Create a new conda environment
conda create -n pspra python=3.9

# Activate the environment
conda activate pspra
```

### Method 2: Using venv (Alternative)

If you don't have conda installed or prefer using Python's built-in virtual environment:

```bash
# Create a new virtual environment
python -m venv pspra_env

# Activate the environment
# On Windows:
pspra_env\Scripts\activate

# On macOS/Linux:
source pspra_env/bin/activate
```

**Note**: Make sure you have Python 3.7 or higher installed. You can check your Python version with:
```bash
python --version
```

### Installing Required Packages

After activating your environment (either conda or venv), install the necessary packages using pip:

```bash
# Core machine learning packages
pip install numpy pandas scikit-learn

# Deep learning and neural networks
pip install tensorflow keras-tcn

# Gradient boosting libraries
pip install lightgbm xgboost catboost

# Additional utilities
pip install joblib scipy
```

Alternative installation using requirements.txt (if you create one):
```bash
pip install -r requirements.txt
```

### Deactivating the Environment

When you're done working with the project:

```bash
# For conda:
conda deactivate

# For venv:
deactivate
```

## Usage

### Running the Training Script

The main training script is `train.py` located in the `script/` directory. Make sure your environment is activated, then navigate to the script directory:

```bash
cd script
python train.py [OPTIONS]
```

### Command Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--experiment` | str | "p1" | Experiment identifier (p1, p2, p3, p3m) |
| `--model_name` | str | "TCN" | Model type to train |
| `--result_path` | str | /results | Directory to save training results |
| `--model_save_path` | str | /models | Directory to save trained models |
| `--overwrite` | flag | False | Overwrite existing results if they exist |

### Experiment Types

- **p1**: Experiment 1 from the paper - Basic PSP risk assessment
- **p2**: Experiment 2 from the paper - Extended temporal analysis
- **p3**: Experiment 3 from the paper - Recent data analysis (univariate)
- **p3m**: Experiment 3 multivariate case - Using multiple toxin compounds

### Available Models

#### Neural Network Models
- **TCN**: Temporal Convolutional Network
- **RNN**: Recurrent Neural Network (LSTM-based)

#### Traditional Machine Learning Models
- **GBC**: Gradient Boosting Classifier
- **RF**: Random Forest
- **LR**: Logistic Regression
- **DT**: Decision Tree
- **catboost**: CatBoost Classifier
- **ADA**: AdaBoost Classifier
- **LDA**: Linear Discriminant Analysis
- **ET**: Extra Trees Classifier
- **LGB**: LightGBM Classifier
- **QDA**: Quadratic Discriminant Analysis
- **NB**: Naive Bayes
- **XGB**: XGBoost Classifier
- **KNN**: K-Nearest Neighbors

#### Ensemble Model
- **stacking_ensemble**: Stacking ensemble combining all above models

### Example Commands

```bash
# Train TCN model for experiment 1
python train.py --experiment p1 --model_name TCN

# Train Random Forest for experiment 3 multivariate
python train.py --experiment p3m --model_name RF

# Train stacking ensemble for experiment 2 with custom paths
python train.py --experiment p2 --model_name stacking_ensemble --result_path /custom/results --model_save_path /custom/models

# Overwrite existing results
python train.py --experiment p1 --model_name RNN --overwrite
```

## Output Files

### Results Directory Structure

When using default paths, the results are organized as follows:

```
results/
├── p1/                   # Experiment 1 results
│   ├── TCN/
│   │   └── results_YYYYMMDD_HHMMSS.npz
│   ├── RNN/
│   └── RF/
├── p2/                   # Experiment 2 results
├── p3/                   # Experiment 3 results
└── p3m/                  # Experiment 3 multivariate results
```

### Results File Content

Each `results_YYYYMMDD_HHMMSS.npz` file contains:

- **X_train**: Training input data
- **y_train**: Training target labels
- **X_test**: Test input data  
- **y_test**: Test target labels
- **y_pred_proba**: Predicted probabilities on test set
- **y_pred**: Binary predictions on test set
- **y_pred_proba_train**: Predicted probabilities on training set
- **y_pred_train**: Binary predictions on training set

### Models Directory Structure

```
models/
├── p1/                   # Experiment 1 models
│   ├── TCN/
│   │   └── best_model_YYYYMMDD_HHMMSS.keras
│   ├── RNN/
│   └── RF/
│       └── model_YYYYMMDD_HHMMSS.joblib
├── p2/                   # Experiment 2 models
├── p3/                   # Experiment 3 models
└── p3m/                  # Experiment 3 multivariate models
```

### Model File Types

- **Neural Network Models** (TCN, RNN): Saved as `.keras` files
- **Traditional ML Models**: Saved as `.joblib` files using scikit-learn format
- **Stacking Ensemble**: Saved as `.joblib` files

## Loading and Using Results

### Loading Results

```python
import numpy as np

# Load results
results = np.load('results/p1/TCN/results_20240101_120000.npz')

# Access predictions
y_test = results['y_test']
y_pred_proba = results['y_pred_proba']
y_pred = results['y_pred']

# Calculate metrics
from sklearn.metrics import roc_auc_score, accuracy_score
auc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)
```

### Loading Trained Models

```python
# For neural network models
from tensorflow import keras
model = keras.models.load_model('models/p1/TCN/best_model_20240101_120000.keras')

# For traditional ML models
import joblib
model = joblib.load('models/p1/RF/model_20240101_120000.joblib')
```

## Data Requirements

The code expects data files in the following format:

### Univariate Data (p1, p2, p3)
- File: `data/data_univariate.csv`
- Required columns: `site`, `date`, `value`, plus other metadata

### Multivariate Data (p3m)
- File: `data/data_multivariate.csv`  
- Required columns: `site`, `date`, `compound`, `value`

## Citation

If you use this code in your research, please cite our paper:

```bibtex
[Citation information to be added when paper is published]
```

## License

[License information to be added]

## Contact

For questions about the code or methodology, please contact [contact information].
