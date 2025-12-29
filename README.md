# 📈 Stock Price Prediction System

A **modular, end‑to‑end stock price prediction framework** designed for **time‑series forecasting**, **feature‑rich experimentation**, and **robust evaluation** using both **machine learning** and **deep learning** models.

This repository emphasizes **correct temporal modeling**, **walk‑forward validation**, and **production‑grade structure**—making it suitable for research, portfolio projects, and real‑world extensions.

---

## 🚀 Project Objectives

- Predict **next‑day stock price / return** using historical and engineered features
- Maintain **strict time‑order integrity** (no data leakage)
- Support **classical ML baselines** and **deep learning models**
- Enable **walk‑forward / rolling‑window evaluation**
- Keep the system **config‑driven, extensible, and reproducible**

---

## 🧠 Modeling Approaches

### Machine Learning
- XGBoost Regressor (primary baseline)
- Designed for:
  - Non‑linear relationships
  - Tabular time‑series features
  - Fast experimentation

### Deep Learning (Planned / In Progress)
- LSTM
- GRU
- Sliding‑window sequence modeling
- Expanding‑window walk‑forward training

---

## 🗂️ Repository Structure

```
Stock-Price-Prediction/
│
├── config/
│   └── config.py               # Central configuration (features, params, paths)
│
├── data/
│   ├── raw/                    # Original merged dataset
│   ├── processed/              # Feature‑engineered datasets
│   └── splits/                 # Train / validation splits (optional)
│
├── features/
│   └── feature_engineering.py  # Price, momentum, volatility indicators
│
├── modeling/
│   ├── train.py                # XGBoost training (walk‑forward capable)
│   ├── evaluate.py             # Metrics + directional accuracy
│   └── deep_models.py          # LSTM / GRU architectures
│
├── utils/
│   ├── metrics.py              # MAE, RMSE, R², direction accuracy
│   └── time_series.py          # Rolling / expanding window utilities
│
├── notebooks/
│   ├── phase_1_eda.ipynb
│   ├── phase_2_ml.ipynb
│   └── phase_3_dl.ipynb
│
├── models/
│   └── xgboost/                # Saved trained models
│
├── results/
│   └── metrics.json            # Evaluation outputs
│
├── README.md
└── requirements.txt
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

⚠️ Calendar features are intentionally excluded in early phases to avoid leakage and overfitting.

---

## 📊 Evaluation Strategy

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

## ⚙️ Configuration‑Driven Design

All critical parameters are centralized in `config/config.py`:

- Feature inclusion/exclusion
- Target column selection
- Model hyperparameter grids
- Paths for models and outputs

This ensures:
- Reproducibility
- Clean experimentation
- Minimal hard‑coding

---

## ▶️ How to Run

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Prepare Dataset
Ensure a time‑ordered dataframe with at least:
```
Date | StockPrice | Data_Value
```

Run feature engineering:
```bash
python features/feature_engineering.py
```

### 3️⃣ Train XGBoost Model
```bash
python modeling/train.py
```

### 4️⃣ Evaluate
```bash
python modeling/evaluate.py
```

---

## 🔮 Roadmap

- [ ] Full LSTM / GRU walk‑forward training
- [ ] Multi‑step forecasting (t+1, t+5)
- [ ] Probabilistic forecasting (prediction intervals)
- [ ] Trading strategy backtesting
- [ ] Model explainability (SHAP)

---

## 🧩 Design Philosophy

✔ Time‑series correctness over convenience  
✔ Strong baselines before complex models  
✔ Research‑friendly yet production‑ready  
✔ Modular, testable, extensible

---

## 📌 Disclaimer

This project is for **educational and research purposes only**.
It does **not constitute financial advice**.

---


