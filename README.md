# 📈 Stock Price Prediction (LSTM & GRU)

Leakage-free deep learning pipeline for **next-day stock price prediction** using **LSTM and GRU** models.

The system predicts **next-day log returns**, then reconstructs prices via inverse transformation — with **all feature engineering and modeling done inside the pipeline**, reading **only raw data**.

---

## 🔑 Key Features
- LSTM & GRU models (no XGBoost)
- Strict time-series causality (no leakage)
- Raw-data-only access (`data/raw/`)
- End-to-end pipeline (`pipeline.py`)
- Automatic metrics & plots generation

---

## 📂 Project Structure
```
StockPricePrediction/
├── pipeline.py          # Main entry point
├── dataset.py           # Raw data loading
├── features.py          # Causal feature engineering
├── plots.py             # Plot generation
├── modeling/
│   └── train_dl.py      # LSTM & GRU training
├── config.py
reports/
├── metrics.txt
└── figures/
    └── lstm_gru_price_predictions.png
```

---

## ▶️ How to Run
```bash
pip install -r requirements.txt
python StockPricePrediction/pipeline.py
```

Ensure raw files exist:
```
data/raw/StockPrice.csv
data/raw/Data.csv
```

---

## 📊 Evaluation
- MAE, RMSE, R² (on reconstructed prices)
- Results saved in `reports/`
- Prediction plot saved in `reports/figures/`

---

## 🧠 Note
Low or negative R² is expected for daily stock data — correctness and causality are prioritized over inflated metrics.

---

## 📜 License
MIT License
