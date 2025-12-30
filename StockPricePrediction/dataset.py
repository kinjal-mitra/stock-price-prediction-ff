import pandas as pd
from config import RAW_DATA_DIR, DATE_COL, PRICE_COL, EXTERNAL_COL

def load_raw_data():
    price_df = pd.read_csv(RAW_DATA_DIR / "StockPrice.csv")
    data_df = pd.read_csv(RAW_DATA_DIR / "Data.csv")

    price_df[DATE_COL] = pd.to_datetime(price_df[DATE_COL])
    data_df[DATE_COL] = pd.to_datetime(data_df[DATE_COL])

    price_df = price_df.sort_values(DATE_COL)
    data_df = data_df.sort_values(DATE_COL)

    df = pd.merge(price_df, data_df, on=DATE_COL, how="inner")
    return df.reset_index(drop=True)
