import pandas as pd
import numpy as np

#For plotting and missing data
import matplotlib.pyplot as plt
import seaborn as sns


#For modelling
from math import sqrt
from scipy.special import logsumexp
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Lasso,
    ElasticNet,
    BayesianRidge
)
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
import tensorflow as tf
from scipy.spatial.distance import cdist #For Euclidean distances
from skopt import BayesSearchCV #For Bayesian selection of hyperparameters
import gower #For distances when variables are categorical and continuos
from pygam import LinearGAM
from sklearn.ensemble import GradientBoostingRegressor #Gradient boosting machine
from sklearn.metrics import mean_absolute_error  # Change the import
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import KFold


from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from typing import Any, Dict, Optional, Tuple



#Models fitting
# --------------------------------------------------------------------
# 1️⃣  A registry that maps a short key → a scikit‑learn estimator
# --------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, Any] = {
    "linear": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "elasticnet": ElasticNet,
    "bayesian": BayesianRidge,
    "svr": SVR,
    "gbm": GradientBoostingRegressor,
    "dt": DecisionTreeRegressor
}

# --------------------------------------------------------------------
# 2️⃣  The core routine – pure modelling + optional hyper‑parameter search
# --------------------------------------------------------------------
def fit_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    *,
    model_name: str,
    param_grid: Optional[Dict[str, Any]] = None,
    search: str = "grid",          # "grid" or "random"
    n_iter: int = 30,              # only used for RandomizedSearchCV
    cv: int = 5,
    scoring: str = "neg_mean_absolute_error",
    n_jobs: int = -1,
    return_metrics: bool = False,
    y_test: Optional[pd.Series] = None,
) -> Tuple[BaseEstimator, np.ndarray, Optional[Dict[str, float]]]:
    """
    Train a model (optionally with GridSearchCV / RandomizedSearchCV) and
    return the fitted estimator + predictions on X_test.

    Parameters
    ----------
    X_train, y_train : pd.DataFrame / pd.Series
        Training data.
    X_test : pd.DataFrame
        Test set (features only).
    model_name : str
        Key in MODEL_REGISTRY - e.g. "ridge".
    param_grid : dict | None
        Hyper-parameter dictionary.  If None and the chosen model
        has tunable parameters, a sensible default is supplied.
    search : {"grid", "random"}
        Which search strategy to use.
    n_iter : int
        Number of iterations for RandomizedSearchCV (ignored for grid).
    cv : int
        Number of CV folds.
    scoring : str
        Scikit-learn scoring string.
    n_jobs : int
        Parallelism (``-1`` = use all cores).
    return_metrics : bool
        If True, the function also prints and returns a dict of
        MAE / RMSE / R² on the *test* set (you need to supply `y_test`
        separately if you want that - see the example below).

    Returns
    -------
    fitted_estimator : sklearn.BaseEstimator
        The best estimator after (optional) hyper-parameter search.
    predictions : np.ndarray
        Predictions on `X_test`.
    metrics : dict | None
        If `return_metrics=True` - a small dict of MAE, RMSE, R².
    """

    # ---- 2a. Pick the estimator class ---------------------------------
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name: {model_name!r}.  "
                         f"Valid keys: {list(MODEL_REGISTRY)}")

    EstimatorCls = MODEL_REGISTRY[model_name]
    estimator: BaseEstimator = EstimatorCls()
    
    # Add random state to estimators that support it
    if hasattr(estimator, 'random_state'):
        estimator.random_state = 42

    # ---- 2b. Default hyper‑parameter grids ---------------------------------
    if param_grid is None:
        if model_name == "linear":
            param_grid = {}                     # nothing to tune
        else:
            # Provide a small, generic grid for the most common models
            if model_name == "ridge":
                param_grid = {"alpha": np.logspace(-2, 2, 10)} # 0.01 to 100
            elif model_name == "lasso":
                param_grid = {"alpha": np.logspace(-4, 0, 20)}
            elif model_name == "elasticnet":
                param_grid = {"alpha": np.logspace(-4, 0, 20),
                              "l1_ratio": np.linspace(0.1, 0.9, 9)}
            elif model_name == "bayesian":
                param_grid = {"alpha_1": [1e-6, 1e-4, 1e-2],
                              "lambda_1": [1e-6, 1e-4, 1e-2]}
            elif model_name == "svr":
                param_grid = {"C": np.logspace(-2, 2, 5),    # 5 values
                              "epsilon": [0.1, 0.2],       # Epsilon parameter for the margin of tolerance
                              "kernel": ['linear', 'rbf']} # Kernel types to try: linear and RBF
            elif model_name == "gbm":
                param_grid = {"n_estimators": [50, 100, 150],
                              "learning_rate": [0.01, 0.1, 0.2],
                              "max_depth": [3, 4, 5]}
            elif model_name == "dt":
                param_grid = {"max_depth": [1, 2, 3, 4, 5],
                              "min_samples_split": [2, 3, 4, 5],
                              "min_samples_leaf": [1, 2, 3, 4, 5],
                              "criterion": ['absolute_error', 'squared_error']}
            else:
                param_grid = {}

    # ---- 2c. Fit with or without a search ------------------------------
    # Create deterministic cross-validation splitter
    cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    if param_grid:        # search *only* if there are parameters to try
        if search == "grid":
            searcher = GridSearchCV(
                estimator,
                param_grid=param_grid,
                cv=cv_splitter,
                scoring=scoring,
                n_jobs=n_jobs,
                return_train_score=False,
            )
        elif search == "random":
            searcher = RandomizedSearchCV(
                estimator,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv_splitter,
                scoring=scoring,
                n_jobs=n_jobs,
                random_state=42,
                return_train_score=False,
            )
        else:
            raise ValueError("search must be either 'grid' or 'random'")

        searcher.fit(X_train, y_train)
        fitted = searcher.best_estimator_
        best_params = searcher.best_params_
        print(f"[{model_name.upper()}] Best params: {best_params}")
    else:                 # No hyper‑parameters – plain fit
        estimator.fit(X_train, y_train)
        fitted = estimator
        best_params = None

    # ---- 2d. Predict on the test set ------------------------------------
    preds = fitted.predict(X_test)

    # ---- 2e. (Optional) compute test‑set metrics ------------------------
    metrics = None
    if return_metrics:
        if y_test is None:
            raise RuntimeError("`return_metrics=True` requires `y_test` "
                               "to be defined in the caller's namespace.")
        metrics = {
            "mae": mean_absolute_error(y_test, preds),
            "rmse": np.sqrt(mean_squared_error(y_test, preds)),
            "r2": r2_score(y_test, preds),
        }

    return fitted, preds, metrics



