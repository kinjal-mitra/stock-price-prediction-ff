import numpy as np
import pandas as pd
from config import PRICE_COL, EXTERNAL_COL, TARGET_COL

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # Log return
    df["log_return"] = np.log(df[PRICE_COL] / df[PRICE_COL].shift(1))

    # Target: next-day log return
    df[TARGET_COL] = df["log_return"].shift(-1)

    # Lagged returns
    for lag in range(1, 11):
        df[f"log_return_lag_{lag}"] = df["log_return"].shift(lag)

    # Rolling statistics
    for w in [5, 10, 20]:
        df[f"return_mean_{w}"] = df["log_return"].rolling(w).mean().shift(1)
        df[f"return_std_{w}"] = df["log_return"].rolling(w).std().shift(1)

    # Price trend
    for w in [10, 20, 50]:
        ma = df[PRICE_COL].rolling(w).mean().shift(1)
        df[f"price_ma_ratio_{w}"] = df[PRICE_COL].shift(1) / ma

    # Volatility
    for w in [5, 10, 20]:
        df[f"volatility_{w}"] = df["log_return"].rolling(w).std().shift(1)

    # External signal (lagged only)
    for lag in [1, 3, 5, 10]:
        df[f"data_lag_{lag}"] = df[EXTERNAL_COL].shift(lag)

    return df.dropna().reset_index(drop=True)
