import numpy as np

def PEHE(y:np.ndarray=None, y_hat:np.ndarray=None)->float:
    """Compute Precision in Estimation of Heterogeneous Effect.

    Args:
    - y:     potential outcomes
    - y_hat: estimated potential outcomes

    Returns:
    - PEHE_val: computed PEHE
    """
    PEHE_val = np.mean( ( (y[:,1] - y[:,0]) - (y_hat[:,1] - y_hat[:,0]) )**2 )
    return PEHE_val


def ATE(y:np.ndarray=None, y_hat:np.ndarray=None)->float:
    """Compute Average Treatment Effect.

    Args:
    - y:     potential outcomes
    - y_hat: estimated potential outcomes

    Returns:
    - ATE_val: computed ATE
    """
    ATE_val = np.abs(np.mean(y[:,1] - y[:,0]) - np.mean(y_hat[:,1] - y_hat[:,0]))
    return ATE_val