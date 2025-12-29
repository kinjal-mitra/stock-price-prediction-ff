# dataset.py

import pandas as pd
from config import (
    RAW_DATA_DIR,
    DATE_COL,
    EXTERNAL_DATA_COL,
    PRICE_COL
)

def load_and_merge_raw_data() -> pd.DataFrame:
    """
    Loads Data.csv and StockPrice.csv, sorts, merges, and cleans.
    """

    data_df = pd.read_csv(RAW_DATA_DIR / "Data.csv")
    price_df = pd.read_csv(RAW_DATA_DIR / "StockPrice.csv")

    # Rename for consistency
    data_df = data_df.rename(columns={"Data": EXTERNAL_DATA_COL})
    price_df = price_df.rename(columns={"Price": PRICE_COL})

    # Parse dates
    data_df[DATE_COL] = pd.to_datetime(data_df[DATE_COL])
    price_df[DATE_COL] = pd.to_datetime(price_df[DATE_COL])

    # Sort ASCENDING (critical for time series)
    data_df = data_df.sort_values(DATE_COL)
    price_df = price_df.sort_values(DATE_COL)

    # Merge
    df = pd.merge(
        data_df,
        price_df,
        on=DATE_COL,
        how="inner"
    ).reset_index(drop=True)

    return df
