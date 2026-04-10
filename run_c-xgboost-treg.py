# %%
"""
Demo script for C-XGBoost

This script demonstrates:
- Loading training and testing data
- Training a C-XGBoost model
- Evaluating treatment effect estimation using ATE and PEHE
"""

import os
import random
import warnings
import numpy as np
from utils.metrics import PEHE, ATE
from models.cxgboost_treg import CXGBoostTreg

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
warnings.filterwarnings("ignore")

SEED = 42
DATA_DIR = "Data"

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
def load_dataset(path: str):
    """Load dataset from a NumPy .npz file."""
    data = np.load(path)
    return data["X"], data["T"], data["Y"], data["potential_Y"]


# Load training data
trainX, trainT, trainY, train_potential_Y = load_dataset(
    os.path.join(DATA_DIR, "train.npz")
)

# Load test data
testX, testT, testY, test_potential_Y = load_dataset(
    os.path.join(DATA_DIR, "test.npz")
)

print("[INFO] Dataset loaded successfully")
print(f"[INFO] Training samples: {trainX.shape[0]}")
print(f"[INFO] Testing samples:  {testX.shape[0]}")

# -----------------------------------------------------------------------------
# Model Configuration and Training
# -----------------------------------------------------------------------------
model_params = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 1e-1,
    "n_jobs": -1,
    "lambda": 5,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "tree_method": "hist",
}

model = CXGBoostTreg(parameters=model_params, beta=1.0)
model.fit(X=trainX, t=trainT, y=trainY)

# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------
predictions = model.predict(testX)

# Stack potential outcome predictions: [Y(0), Y(1)]
predicted_potential_outcomes = np.column_stack(
    [predictions["y0_hat"], predictions["y1_hat"]]
)

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
# Ground-truth ATE
true_ate = (test_potential_Y[:, 1] - test_potential_Y[:, 0]).mean()

# Estimated errors
ate_error = ATE(test_potential_Y, predicted_potential_outcomes)
pehe_score = PEHE(test_potential_Y, predicted_potential_outcomes)

print("\n\n[RESULTS]")
print(f"True ATE     : {true_ate:.3f}")
print(f"ATE Error    : {ate_error:.3f}")
print(f"PEHE         : {pehe_score:.3f}")