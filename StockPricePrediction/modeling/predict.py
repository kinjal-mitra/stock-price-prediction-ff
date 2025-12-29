# StockPricePrediction/modeling/predict.py

import joblib
import numpy as np
from config import MODEL_DIR, DROP_COLS, TARGET_COL

def load_model():
    return joblib.load(MODEL_DIR / "xgboost_model.pkl")

def predict(df):
    model = load_model()
    X = df.drop(columns=DROP_COLS + [TARGET_COL]).values
    return model.predict(X)
