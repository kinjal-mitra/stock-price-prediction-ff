# StockPricePrediction/modeling/train.py

import itertools
import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from config import (
    TARGET_COL,
    DROP_COLS,
    MODEL_DIR,
    XGB_BASE_PARAMS,
    XGB_PARAM_GRID
)

def train_xgboost(df):
    """
    Train XGBoost with time-ordered train/validation split.
    Tracks RMSE (selection metric), MAE, and R².
    """

    X = df.drop(columns=DROP_COLS + [TARGET_COL]).values
    y = df[TARGET_COL].values

    split_idx = int(len(df) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    best_rmse = np.inf
    best_mae = None
    best_r2 = None
    best_model = None

    for params in itertools.product(*XGB_PARAM_GRID.values()):
        param_dict = dict(zip(XGB_PARAM_GRID.keys(), params))

        model = XGBRegressor(**XGB_BASE_PARAMS, **param_dict)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)

        if rmse < best_rmse:
            best_rmse = rmse
            best_mae = mae
            best_r2 = r2
            best_model = model

    # Save best model
    model_path = MODEL_DIR / "xgboost_model.pkl"
    joblib.dump(best_model, model_path)

    print("XGBOOST VALIDATION PERFORMANCE")
    print(f"RMSE : {best_rmse:.6f}")
    print(f"MAE  : {best_mae:.6f}")
    print(f"R²   : {best_r2:.4f}")
    print(f"\n Model saved to {model_path}")

    return best_model
