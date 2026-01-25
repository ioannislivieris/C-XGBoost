import time
import xgboost
import numpy as np
from typing import Dict, Optional
from sklearn.ensemble import RandomForestClassifier


class CXGBoost:
    """
    Causal XGBoost model that jointly estimates potential outcomes Y(0) and Y(1)
    using DragonNet-style treatment masking.

    Supports both binary and multiclass classification tasks.
    """

    def __init__(
        self, 
        parameters: Optional[dict] = None, 
        propensity_model=None,
        scale_pos_weight: Optional[float] = None,
    ) -> None:
        """
        Initialize the CXGBoost model.
        
        Args:
            parameters: XGBoost hyperparameters (dict)
            propensity_model: Sklearn-style classifier for propensity scores
            scale_pos_weight: Weight for positive class in weighted_bce (binary only)
        """
        self.parameters = parameters or {}        
        self.model = xgboost.XGBRegressor(**self.parameters)            
        
        # Initialize propensity score model
        self.propensity_model = propensity_model or RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
        )

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the causal model and the propensity score model.
        
        Args:
            X: Covariates, shape (n_samples, n_features)
            t: Treatment indicator, shape (n_samples,), binary 0/1
            y: Outcome, shape (n_samples,)
                - For binary task: binary 0/1
                - For multiclass task: integer class labels 0, 1, ..., num_classes-1
        """
        # Create treatment indicator mask (DragonNet style)
        tt = np.array([[1, 0] if t_i == 0 else [0, 1] for t_i in t]).flatten()
        
        custom_loss = self._get_multiclass_loss(tt)
        
        # Configure XGBoost
        self.model.set_params(
            objective=custom_loss,
            num_target=2,
            multi_strategy="multi_output_tree"
        )
        
        # Prepare multi-output target: [y, y] for both Y(0) and Y(1)
        yt = np.column_stack([y, y])
        
        # Start timer
        start = time.time()
        # Fit the XGBoost model
        print("[INFO] Fitting C-XGBoost...")
        self.model.fit(X, yt)
        # Fit the propensity score model
        print("[INFO] Fitting propensity model...")
        self.propensity_model.fit(X, t)
        print('[INFO] Model training completed - Time %.2f seconds' % (time.time() - start))
        
    def _get_multiclass_loss(self, tt: np.ndarray):
        """Get MSE loss for multiclass classification task."""
        
        def custom_loss(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
            """Mean squared error with treatment masking for multiclass."""
            grad = 2 * (y_pred.flatten() - y_true.flatten()) * tt
            hess = np.full_like(y_pred.flatten(), 2.0) * tt
            hess = np.maximum(hess, 1e-16)
            
            return grad, hess
        
        return custom_loss

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate potential outcome predictions and propensity scores.
        
        Args:
            X: Covariates, shape (n_samples, n_features)
            
        Returns:
            Dictionary with:
                - 'y0_hat': Predicted outcome under control (T=0)
                    Binary: probabilities in [0, 1]
                    Multiclass: class probabilities, shape (n_samples, num_classes)
                - 'y1_hat': Predicted outcome under treatment (T=1)
                    Binary: probabilities in [0, 1]
                    Multiclass: class probabilities, shape (n_samples, num_classes)
                - 'propensity_score': P(T=1|X)
        """
        pred = self.model.predict(X)
        y0_hat = pred[:, 0]  # Y_0 prediction
        y1_hat = pred[:, 1]  # Y_1 prediction
        
           
        return {
            'y0_hat': y0_hat,
            'y1_hat': y1_hat,
            'propensity_score': self.propensity_model.predict_proba(X)[:, 1]
        }