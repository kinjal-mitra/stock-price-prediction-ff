# config.py

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

PROCESSED_DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Columns
DATE_COL = "Date"
EXTERNAL_DATA_COL = "Data"
PRICE_COL = "StockPrice"
TARGET_COL = "log_return"

DROP_COLS = [DATE_COL, PRICE_COL]


#XGBOOST Config

XGB_BASE_PARAMS = {
    "objective": "reg:squarederror",
    "random_state": 42,
    "verbosity": 0
}

XGB_PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [3, 5],
    "learning_rate": [0.03, 0.05],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

'''
print(PROJECT_ROOT)
print(DATA_DIR)
print(RAW_DATA_DIR)
print(PROCESSED_DATA_DIR)
print(MODEL_DIR)
'''