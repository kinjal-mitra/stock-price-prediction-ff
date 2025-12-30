#  Stock Price Prediction System

A **modular, end‑to‑end stock price prediction framework** designed for **time‑series forecasting**, **feature‑rich experimentation**, and **robust evaluation** using both **machine learning** and **deep learning** models.

This repository emphasizes **correct temporal modeling**, **walk‑forward validation**, and **production‑grade structure**—making it suitable for research, portfolio projects, and real‑world extensions.

---

##  Project Objectives

- Predict **next‑day stock price / return** using historical and engineered features
- Maintain **strict time‑order integrity** (no data leakage)
- Support **classical ML baselines** and **deep learning models**
- Enable **walk‑forward / rolling‑window evaluation**
- Keep the system **config‑driven, extensible, and reproducible**

---

##  Modeling Approaches

### 1. Machine Learning
- XGBoost Regressor (primary baseline)
- Designed for:
  - Non‑linear relationships
  - Tabular time‑series features
  - Fast experimentation

### 2. Deep Learning (Planned / In Progress)
- LSTM
- GRU
- Sliding‑window sequence modeling
- Expanding‑window walk‑forward training

---

##  Repository Structure

```
stock-price-prediction-ff/
├── data/
│   ├── external/
│   ├── interim/
│   │   └── features_dataset.csv (1.3MB)
│   ├── processed/
│   │   ├── splits/
│   │   │   ├── X_test.npy (2.8MB)
│   │   │   ├── X_train.npy (12.9MB)
│   │   │   ├── X_val.npy (2.8MB)
│   │   │   ├── y_test.npy (4.5KB)
│   │   │   ├── y_train.npy (20.5KB)
│   │   │   └── y_val.npy (4.5KB)
│   │   ├── processed_dataset.csv (818.2KB)
│   │   ├── X_features.npy (762.5KB)
│   │   └── y_target.npy (29.4KB)
│   └── raw/
│       ├── Data.csv (70.0KB)               # Provided Raw Data File
│       └── StockPrice.csv (73.1KB)         # Provided Raw Stock Price file
├── docs/
├── models/
│   └── xgboost_model.pkl (316.6KB)
├── notebooks/
│   ├── 1.EDA.ipynb (190.9KB)
│   ├── 2.CreateFeatures.ipynb.ipynb (27.7KB)
│   ├── 3.FeatureEngineering.ipynb (8.8KB)
│   ├── 4.TraningModel.ipynb (41.5KB)
│   ├── 5.TrainingXGBoostModel.ipynb (28.3KB)
│   ├── best_gru_model.pt (169.1KB)
│   └── best_lstm_model.pt (865.1KB)
├── references/
│   └── feature_scaler.pkl (1.7KB)
├── reports/
│   └── figures/
│       ├── data_timeseries.png (28.8KB)
│       ├── data_vs_price_dual_axis.png (54.4KB)
│       ├── lstm_vs_gru_predictions.png (59.5KB)
│       └── price_timeseries.png (39.3KB)
├── StockPricePrediction/
│   ├── modeling/
│   │   ├── __init__.py (0.0B)
│   │   ├── predict.py (345.0B)
│   │   └── train.py (1.7KB)
│   ├── __init__.py (55.0B)
│   ├── config.py (904.0B)
│   ├── data_processing.py (269.0B)
│   ├── dataset.py (994.0B)
│   ├── features.py (1.6KB)
│   ├── pipeline.py (481.0B)
│   └── plots.py (803.0B)
├── LICENSE (1.1KB)
├── Makefile (2.5KB)
├── pyproject.toml (772.0B)
├── README.md (5.0KB)
└── requirements.txt (53.0B)
```

---

## 🧪 Feature Engineering

The system constructs **domain‑aware financial indicators**, including:

### Price‑Based Features
- `daily_return`
- `log_return` *(primary ML target)*
- `price_change`
- `price_lag_1` → `price_lag_5`

### Trend & Momentum
- Moving averages: `MA_7`, `MA_30`, `MA_50`
- `price_to_MA7_ratio`
- Momentum: `momentum_5d`, `momentum_20d`

### Volatility & Ranges
- `volatility_7d`
- `bollinger_upper`, `bollinger_lower`
- `rolling_max_20d`, `rolling_min_20d`

### Technical Indicators
- RSI (14)
- MACD + Signal Line

Note: Calendar features are intentionally excluded in early phases to avoid leakage and overfitting.

---

## Evaluation Strategy

### Offline Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- **Directional Accuracy** (up/down correctness)

### Walk‑Forward Validation
- Expanding window training
- Rolling prediction windows
- Realistic simulation of live trading deployment

Example output:
```
WALK‑FORWARD RESULTS
MAE  : 0.00622 ± 0.00213
RMSE : 0.00868 ± 0.00367
R²   : 0.3098
Directional Accuracy: 71.06%
```

---

## Configuration‑Driven Design

All critical parameters are centralized in `StockPricePrdiction/config.py`:

- Feature inclusion/exclusion
- Target column selection
- Model hyperparameter grids
- Paths for models and outputs

This ensures:
- Reproducibility
- Clean experimentation
- Minimal hard‑coding

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
Ensure a time‑ordered dataframe with at least:
```
Date | StockPrice | Data_Value
```


### 3. Train XGBoost Model
```bash
python StockPricePrdiction/modeling/train.py
```

### 4. Evaluate
```bash
python StockPricePrdiction/modeling/evaluate.py
```

OR
### Run Entire Pipeline
```bash
python StockPricePrdiction/pipeline.py
```


---

##  Design Philosophy

- Time‑series correctness over convenience  
- Strong baselines before complex models  
- Research‑friendly yet production‑ready  
- Modular, testable, extensible

---

##  Disclaimer

This project is for **educational and research purposes only**.
It does **not constitute financial advice**.

---


