# data_processing.py

from config import PROCESSED_DATA_DIR

def save_processed_data(df, filename="processed_dataset.csv"):
    path = PROCESSED_DATA_DIR / filename
    df.to_csv(path, index=False)
    print(f" Processed data saved to {path}")
    return path
