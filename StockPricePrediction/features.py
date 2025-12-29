# features.py

import numpy as np
import pandas as pd
from config import PRICE_COL, EXTERNAL_DATA_COL

def create_features(df: pd.DataFrame) -> pd.DataFrame:

    # -----------------------------
    # Returns
    # -----------------------------
    df["daily_return"] = df[PRICE_COL].pct_change()
    df["log_return"] = np.log(df[PRICE_COL] / df[PRICE_COL].shift(1))
    df["price_change"] = df[PRICE_COL].diff()

    # -----------------------------
    # Moving Averages
    # -----------------------------
    for w in [7, 30, 50]:
        df[f"MA_{w}"] = df[PRICE_COL].rolling(w).mean()

    df["price_to_MA7_ratio"] = df[PRICE_COL] / df["MA_7"]

    # -----------------------------
    # Volatility
    # -----------------------------
    df["volatility_7d"] = df["daily_return"].rolling(7).std()

    # -----------------------------
    # Momentum
    # -----------------------------
    df["momentum_5d"] = df[PRICE_COL] - df[PRICE_COL].shift(5)
    df["momentum_20d"] = df[PRICE_COL] - df[PRICE_COL].shift(20)

    # -----------------------------
    # Lagged Prices
    # -----------------------------
    for lag in range(1, 6):
        df[f"price_lag_{lag}"] = df[PRICE_COL].shift(lag)

    # -----------------------------
    # External data lags (important!)
    # -----------------------------
    for lag in [1, 3, 5]:
        df[f"data_lag_{lag}"] = df[EXTERNAL_DATA_COL].shift(lag)

    # -----------------------------
    # Cleanup
    # -----------------------------
    df = df.dropna().reset_index(drop=True)
    return df
