import warnings
warnings.filterwarnings('ignore')

import os
import random
import time
import xgboost
import numpy as np
from utils.metrics import PEHE, ATE

# Random generators initialization
seed=42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)

# %%
# Start timer
start = time.time()

# Load train data
data = np.load('Data/train.npz')
trainX, trainT, trainY, train_potential_Y = data['X'], data['T'], data['Y'], data['potential_Y']
# Load valid data
data = np.load('Data/test.npz')
testX, testT, testY, test_potential_Y = data['X'], data['T'], data['Y'], data['potential_Y']
print("[INFO] Dataset imported")
print(f"[INFO] Number of training instances: {trainX.shape[0]}")
print(f"[INFO] Number of testing instances: {testX.shape[0]}")

    

# Loss function
# -----------------------------------------------------------------------------------------------------
treatment = np.array([[1,0] if x == 0 else [0,1] for x in trainT]).flatten()

def custom_loss(y_true:np.ndarray=None, y_pred:np.ndarray=None)->(np.ndarray, np.ndarray):
    grad = 2*(y_pred.flatten() - y_true.flatten()) * treatment
    hess = (0*y_pred.flatten() + 2)  * treatment

    return grad, hess
    


# Setup XGBoost
# -----------------------------------------------------------------------------------------------------
model = xgboost.XGBRegressor(n_estimators=500, 
                                max_depth=4, 
                                objective=custom_loss, 
                                learning_rate=1e-2, 
                            #  booster="gblinear",
                                n_jobs=-1,
                                tree_method="hist", 
                                multi_strategy="multi_output_tree")
    
# Create outputs for DragonNet (concatenate Y & T)
yt_train = np.concatenate([trainY.reshape(-1,1), trainT.reshape(-1,1)], axis = 1)


# Train model
model.fit(trainX, yt_train, eval_set=[(trainX, train_potential_Y), (testX, test_potential_Y)], verbose=25);
print('[INFO] Model trained')    
print('[INFO] Time %.2f seconds' % (time.time() - start))

# %%
# Get predictions
# -----------------------------------------------------------------------------------------------------
test_y_hat = model.predict(testX)

# Calculate performance metrics
# -----------------------------------------------------------------------------------------------------
# ATE
real_ATE = ( test_potential_Y[:,1] - test_potential_Y[:,0] ).mean()      
# Error ATE
Error_ATE = ATE(test_potential_Y, test_y_hat)  
# PEHE
PEHE_score = PEHE(test_potential_Y, test_y_hat)

print(f"Error_ATE: {Error_ATE:.3f}")
print(f"PEHE: {PEHE_score:.3f}")


