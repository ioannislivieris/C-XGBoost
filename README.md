# C-XGBoost: A tree boosting model for causal effect estimation

This is a repository for the implementation of Causal eXtreme Gradient Boosting (C-XGBoost) model, for causal effect estimation.


**Abstract:** Causal effect estimation aims at estimating the Average Treatment Effect as well as the Conditional Average Treatment Effect of a treatment to an outcome from the available data. This knowledge is important in many safety-critical domains, where it often needs to be extracted from observational data. In this work, we propose a new causal inference model, named C-XGBoost, for the prediction of potential outcomes. The motivation of our approach is to exploit the superiority of tree-based models for handling tabular data together with the notable property of causal inference neural network-based models to learn representations that are useful for estimating the outcome for both the treatment and non-treatment cases. The proposed model also inherits the considerable advantages of XGBoost model such as efficiently handling features with missing values requiring minimum preprocessing effort, as well as it is equipped with regularization techniques to avoid overfitting/bias. Furthermore, we propose a new loss function for efficiently training the proposed causal inference model. The experimental analysis, which is based on the performance profiles of Dolan and Moré as well as on post-hoc and non-parametric statistical tests, provide strong evidence about the effectiveness of the proposed approach.

**Keywords:** Causal inference, XGBoost, treatment effect estimation, potential outcomes

**Cite:** Kiriakidou, N., Livieris I.E. & Diou, C. (2024). [C-XGBoost: A tree boosting model for causal effect estimation](https://link.springer.com/chapter/10.1007/978-3-031-63219-8_5). IFIP International Conference on Artificial Intelligence Applications and Innovations.

<br/>

## Table of contents

- [Features](#features)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to run](#how-to-run)
- [Data Format](#data-format)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Citation](#citation)
- [License](#license)
- [Contact](#mailbox-contact)

<br/>

## Features

- **DragonNet-style treatment masking** for joint estimation of potential outcomes
- **XGBoost-based architecture** leveraging tree boosting for tabular data
- **Propensity score modeling** using Random Forest with balanced class weights
- **Custom loss function** for causal effect estimation with treatment masking
- **Built-in missing value handling** inherited from XGBoost
- **Regularization techniques** $L_1$ and $L_2$  to prevent overfitting
- **Comprehensive evaluation** using PEHE and ATE metrics

<br/>

## Model Architecture

C-XGBoost combines the strengths of tree-based models with causal inference principles:

1. **Multi-output XGBoost regressor**: Simultaneously predicts potential outcomes $y_0$ and $y_1$
2. **Treatment masking**: DragonNet-inspired approach that masks gradients based on observed treatment
3. **Propensity score estimation**: Random Forest classifier for estimating treatment assignment probabilities
4. **Custom MSE loss**: Modified mean squared error with treatment-aware gradient computation

The model estimates:
- $y_0(x)$: Expected outcome under control condition
- $y_1(x)$: Expected outcome under treatment condition  
- $p(x)$: $P(T=1|X)$ for each sample

<br/>

## Project Structure

```
.
├── c-xgboost.py          # Demo script for training and evaluation
├── cxgboost.py           # Core CXGBoost model implementation
├── metrics.py            # Evaluation metrics (PEHE, ATE)
├── requirements.txt      # Python dependencies
├── LICENSE               # License file
├── README.md            # This file
└── Data/                # Data directory (not included)
    ├── train.npz        # Training dataset
    └── test.npz         # Testing dataset
```

<br/>

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Dependencies

The project requires the following packages:
- numpy==2.0.0
- xgboost==2.0.0
- scikit-learn==1.5.0

<br/>

## How to run

1. Create a virtual environment 
```bash
python -m venv .venv
```

2. Activate the virtual environment 
```bash
source .venv/bin/activate
```

3. Install requirements 
```bash
pip install -r requirements.txt
```

4. Run
```bash
python c-xgboost.py
```

<br/>

## Data Format

The model expects data in NumPy `.npz` format with the following structure:

```python
{
    'X': np.ndarray,           # Covariates, shape (n_samples, n_features)
    'T': np.ndarray,           # Treatment indicator, shape (n_samples,), binary 0/1
    'Y': np.ndarray,           # Observed outcome, shape (n_samples,)
    'potential_Y': np.ndarray  # Ground truth potential outcomes, shape (n_samples, 2)
                               # Column 0: Y(0), Column 1: Y(1)
}
```

**Note**: `potential_Y` is only required for evaluation (typically available in simulation studies).

<br/>

## Usage

### Basic Usage

```python
from cxgboost import CXGBoost
import numpy as np

# Load your data
data = np.load('Data/train.npz')
X_train, T_train, Y_train = data['X'], data['T'], data['Y']

# Initialize model
model_params = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.01,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
}

model = CXGBoost(parameters=model_params)

# Train the model
model.fit(X=X_train, t=T_train, y=Y_train)

# Make predictions
test_data = np.load('Data/test.npz')
predictions = model.predict(test_data['X'])

# Access predictions
y0_hat = predictions['y0_hat']              # Control outcome predictions
y1_hat = predictions['y1_hat']              # Treatment outcome predictions
propensity = predictions['propensity_score'] # Treatment propensity scores
```

### Custom Propensity Model

```python
from sklearn.linear_model import LogisticRegression

# Use custom propensity model
custom_propensity = LogisticRegression(max_iter=1000, class_weight='balanced')
model = CXGBoost(parameters=model_params, propensity_model=custom_propensity)
```

### Model Parameters

Key hyperparameters for `CXGBoost`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_estimators` | Number of boosting rounds | 100 |
| `max_depth` | Maximum tree depth | 5 |
| `learning_rate` | Step size shrinkage | 0.01 |
| `reg_alpha` | L1 regularization | 1.0 |
| `reg_lambda` | L2 regularization | 1.0 |
| `tree_method` | Tree construction algorithm | 'hist' |
| `n_jobs` | Number of parallel threads | -1 |

<br/>

## Evaluation Metrics

The implementation includes two standard causal inference metrics:

### PEHE (Precision in Estimation of Heterogeneous Effect)

Measures the accuracy of individual treatment effect estimation:

```python
from metrics import PEHE

pehe_score = PEHE(y_true=potential_outcomes, y_hat=predicted_outcomes)
```

**Formula**: $\text{PEHE} = E[(\tau(x) - \hat{\tau}(x))^2]$, where $\tau(x) = y_1(x) - y_0(x)$

### ATE (Average Treatment Effect)

Measures the error in estimating the population-level treatment effect:

```python
from metrics import ATE

ate_error = ATE(y_true=potential_outcomes, y_hat=predicted_outcomes)
```

**Formula**: $|E[y_1(x) - y_0(x)] - E[\hat{y}_1(x) - \hat{y}_0(x)]|$

<br/>

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{kiriakidou2024c,
  title={C-XGBoost: A tree boosting model for causal effect estimation},
  author={Kiriakidou, Niki and Livieris, Ioannis E and Diou, Christos},
  booktitle={IFIP International Conference on Artificial Intelligence Applications and Innovations},
  pages={58--70},
  year={2024},
  organization={Springer}
}
```

<br/>

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<br/>

## :mailbox: Contact

**Ioannis E. Livieris**  
Email: livieris@uop.gr

---

For questions, issues, or contributions, please open an issue on the repository or contact the authors directly.