# C-XGBoost: Tree Boosting Models for Causal Effect Estimation

This repository contains the implementation of two tree-based causal inference models:

- **C-XGBoost** — a multi-output XGBoost model with DragonNet-style treatment masking for joint estimation of potential outcomes *(AIAI 2024)*
- **C-XGBoost (treg)** — an extended version augmented with a third output head for propensity score estimation and a targeted regularization loss function grounded in non-parametric estimation theory *(Neural Computing & Applications, 2025)*

---

## Table of Contents

- [Overview](#overview)
- [Model Architectures](#model-architectures)
  - [C-XGBoost](#c-xgboost)
  - [C-XGBoost (treg)](#c-xgboost-treg)
- [Theoretical Properties](#theoretical-properties)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Data Format](#data-format)
- [Usage](#usage)
- [Hyperparameters](#hyperparameters)
- [Evaluation Metrics](#evaluation-metrics)
- [Experimental Results](#experimental-results)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Overview

Accurate estimation of treatment effects from observational data is a critical challenge in medicine, economics, and healthcare. Both models in this repository exploit the complementary strengths of two paradigms:

- **Tree-based models** (XGBoost): robust handling of tabular data, resistance to uninformative features, ability to capture irregular/non-smooth patterns, built-in missing value support, and regularization
- **Neural network-based causal inference** (DragonNet): shared representation learning across treatment and control groups through a multi-head architecture

C-XGBoost (treg) further introduces a loss function derived from non-parametric estimation theory that directly enforces the efficient influence curve of the ATE functional, yielding double robustness and semiparametric efficiency guarantees.

---

## Model Architectures

### C-XGBoost

At each boosting iteration, a single multi-output tree is constructed with **two values per leaf** — one for each potential outcome:

- $\hat{Q}_0(\mathbf{x}_i;\theta)$ — predicted outcome under control ($T=0$)
- $\hat{Q}_1(\mathbf{x}_i;\theta)$ — predicted outcome under treatment ($T=1$)

**Training objective** — treatment-masked MSE loss:

$$\mathcal{L}(\mathbf{X};\theta) = \frac{1}{n}\sum_i \left[(1-t_i)(y_i - Q_0)^2 + t_i(y_i - Q_1)^2\right]$$

Gradients are masked via a DragonNet-inspired treatment indicator, ensuring each output head receives gradient signal only from the corresponding group.

Propensity scores are estimated separately via a Random Forest classifier with balanced class weights.

### C-XGBoost (treg)

Extends C-XGBoost along two axes:

**Architecture**: a single multi-output tree with **three values per leaf**:

- $\hat{Q}_0(\mathbf{x}_i;\theta)$ — predicted outcome under control
- $\hat{Q}_1(\mathbf{x}_i;\theta)$ — predicted outcome under treatment
- $z(\mathbf{x}_i;\theta)$ — propensity logit, from which $\hat{g}(\mathbf{x}_i;\theta) = \sigma(z)$

**Training objective** — targeted regularization loss:

$$\mathcal{L}_\varepsilon = \frac{1}{n}\sum_i\left[R_i + \alpha\, C(\hat{g}_i, t_i)\right] + \frac{\beta}{n}\sum_i r_i$$

where $R_i$ is the treatment-masked MSE, $C$ is the cross-entropy loss on the propensity head, and the targeted regularization term is:

$$r_i = (1-t_i)(y_i - \tilde{Q}_0^i)^2 + t_i(y_i - \tilde{Q}_1^i)^2$$

with shifted outcomes:

$$\tilde{Q}_0 = Q_0 - \frac{\varepsilon}{1 - \hat{g}}, \qquad \tilde{Q}_1 = Q_1 + \frac{\varepsilon}{\hat{g}}$$

The shared tree structure at each iteration jointly optimizes all three outputs via a composite gain criterion, making it computationally more efficient than three independent ensembles.

---

## Theoretical Properties

Under standard causal assumptions (unconfoundedness, positivity, SUTVA, regularity), minimizing $\mathcal{L}_\varepsilon$ forces the model to satisfy the non-parametric estimating equation defined by the **efficient influence curve** of the ATE. This yields two formal guarantees for the ATE estimator $\hat{\psi} = \frac{1}{n}\sum_i(\hat{Q}_1^{\text{treg}}(\mathbf{x}_i) - \hat{Q}_0^{\text{treg}}(\mathbf{x}_i))$:

- **Double robustness & fast convergence**: $\hat{\psi}$ converges to the true ATE even when the outcome models and propensity score converge slowly — each can partially compensate for the other's misspecification.
- **Semiparametric efficiency**: $\hat{\psi}$ achieves the lowest possible asymptotic variance among all consistent estimators of the ATE.

---

## Project Structure

```
.
├── models/
│   ├── cxgboost.py           # C-XGBoost implementation
│   └── cxgboost_treg.py      # C-XGBoost (treg) implementation
├── utils/
│   └── metrics.py            # Evaluation metrics (PEHE, ATE)
├── run_c-xgboost.py          # Demo script for C-XGBoost
├── run_c-xgboost-treg.py     # Demo script for C-XGBoost (treg)
├── requirements.txt          # Python dependencies
├── LICENSE
├── README.md
└── Data/                     # Data directory (not included)
    ├── train.npz
    └── test.npz
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Dependencies

```
numpy>=2.0.0
xgboost>=2.0.0
scikit-learn>=1.5.0
```

### Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## How to Run

**C-XGBoost:**
```bash
python run_c-xgboost.py
```

**C-XGBoost (treg):**
```bash
python run_c-xgboost-treg.py
```

---

## Data Format

Both models expect NumPy `.npz` files with the following keys:

```python
{
    'X':           np.ndarray,  # Covariates,           shape (n, d)
    'T':           np.ndarray,  # Treatment indicator,  shape (n,), binary 0/1
    'Y':           np.ndarray,  # Observed outcome,     shape (n,)
    'potential_Y': np.ndarray   # Ground-truth outcomes, shape (n, 2)
                                # col 0: Y(0),  col 1: Y(1)
}
```

> `potential_Y` is required only for evaluation and is typically available in semi-synthetic benchmark settings.

---

## Usage

### C-XGBoost

```python
from models.cxgboost import CXGBoost
import numpy as np

data = np.load('Data/train.npz')
X_train, T_train, Y_train = data['X'], data['T'], data['Y']

model = CXGBoost(parameters={
    'n_estimators':  100,
    'max_depth':     5,
    'learning_rate': 1e-2,
    'reg_alpha':     1.0,
    'reg_lambda':    1.0,
    'tree_method':   'hist',
    'n_jobs':        -1,
})

model.fit(X=X_train, t=T_train, y=Y_train)

test  = np.load('Data/test.npz')
preds = model.predict(test['X'])

y0_hat        = preds['y0_hat']           # Predicted Y(0)
y1_hat        = preds['y1_hat']           # Predicted Y(1)
propensity    = preds['propensity_score'] # P(T=1|X) from Random Forest
```

#### Custom propensity model

```python
from sklearn.linear_model import LogisticRegression

model = CXGBoost(
    parameters=model_params,
    propensity_model=LogisticRegression(max_iter=1000, class_weight='balanced')
)
```

---

### C-XGBoost (treg)

```python
from models.cxgboost_treg import CXGBoostTreg
import numpy as np

data = np.load('Data/train.npz')
X_train, T_train, Y_train = data['X'], data['T'], data['Y']

model = CXGBoostTreg(
    parameters={
        'n_estimators':  200,
        'max_depth':     4,
        'learning_rate': 1e-1,
        'reg_alpha':     1.0,
        'reg_lambda':    1.0,
        'tree_method':   'hist',
        'n_jobs':        -1,
    },
    alpha=1.0,    # Weight for propensity cross-entropy term
    beta=1.0,     # Weight for targeted regularization term
    epsilon=0.01, # Targeted regularization shift parameter
)

model.fit(X=X_train, t=T_train, y=Y_train)

test  = np.load('Data/test.npz')
preds = model.predict(test['X'])

y0_hat     = preds['y0_hat']           # Predicted Y(0)
y1_hat     = preds['y1_hat']           # Predicted Y(1)
propensity = preds['propensity_score'] # P(T=1|X) = sigmoid(z)
```

---

## Hyperparameters

### XGBoost parameters (both models)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_estimators` | Number of boosting rounds | 100 |
| `max_depth` | Maximum tree depth | 5 |
| `learning_rate` | Step size shrinkage | 0.01 |
| `reg_alpha` | L1 regularization | 1.0 |
| `reg_lambda` | L2 regularization | 1.0 |
| `tree_method` | Tree construction algorithm | `'hist'` |
| `n_jobs` | Parallel threads | -1 |

### C-XGBoost (treg) specific parameters

| Parameter | Description | Recommended range |
|-----------|-------------|-------------------|
| `alpha` | Weight for propensity cross-entropy | 1.0 |
| `beta` | Weight for targeted regularization | 0.4–1.0 |
| `epsilon` | Targeted regularization shift | 0.01 |

> **Tuning `beta`**: Ablation experiments indicate that for complex or imbalanced datasets (e.g., IHDP, ACIC) values of `beta` ∈ {0.8, 1.0} work best, while moderate values around 0.4–0.6 are preferable for simpler datasets. Setting `beta=0` recovers the base C-XGBoost loss (without targeted regularization).

---

## Evaluation Metrics

### PEHE (Precision in Estimation of Heterogeneous Effect)

$$\text{PEHE} = \frac{1}{n}\sum_i\left[(\tau(\mathbf{x}_i) - \hat{\tau}(\mathbf{x}_i))^2\right], \quad \tau(\mathbf{x}) = y_1(\mathbf{x}) - y_0(\mathbf{x})$$

```python
from utils.metrics import PEHE
pehe = PEHE(y_true=potential_outcomes, y_hat=predicted_outcomes)
```

### ATE error

$$|\epsilon_{\text{ATE}}| = \left|\frac{1}{n}\sum_i(y_1(\mathbf{x}_i) - y_0(\mathbf{x}_i)) - \frac{1}{n}\sum_i(\hat{y}_1(\mathbf{x}_i) - \hat{y}_0(\mathbf{x}_i))\right|$$

```python
from utils.metrics import ATE
ate_error = ATE(y_true=potential_outcomes, y_hat=predicted_outcomes)
```

---

## Experimental Results

Both models were evaluated on three semi-synthetic benchmark collections (Synthetic, IHDP, ACIC) using non-parametric Friedman Aligned-Ranks (FAR) test and Li post-hoc test (α = 5%). Competitors include R-Forest, C-Forest, DragonNet, and MINT-DragonNet.

**C-XGBoost (treg) ranked first overall** across all three collections and both metrics. Key findings:

- On **Synthetic**: C-XGBoost (treg) significantly outperforms all competitors on both $|\epsilon_{\text{ATE}}|$ and PEHE.
- On **IHDP**: C-XGBoost (treg) matches MINT-DragonNet (best neural network model) and significantly outperforms the base C-XGBoost, demonstrating the impact of targeted regularization on imbalanced data.
- On **ACIC**: C-XGBoost (treg) leads, followed by tree-based models; neural network-based models report poor performance on this collection.

Full statistical tables are provided in the respective papers.

---

## Citation

If you use this code, please cite the relevant paper(s):

**C-XGBoost** (AIAI 2024):
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

**C-XGBoost (treg)** (Neural Computing & Applications, 2025):
```bibtex
@article{kiriakidou2025,
  title={C-XGBoost model with targeted regularization for treatment effect estimation},
  author={Kiriakidou, Niki and Livieris, Ioannis E and Diou, Christos},
  journal={Neural Computing \& Applications},
  year={2025},
  publisher={Springer}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Contact

**Ioannis E. Livieris** — livieris@uop.gr  
**Niki Kiriakidou** — kiriakidou@hua.gr  
**Christos Diou** — cdiou@hua.gr

For questions or issues, please open an issue on the repository or contact the authors directly.