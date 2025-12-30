# 📘 Project Documentation  
## Stock Price Prediction using LSTM & GRU (Leakage-Free Pipeline)

---

## 1. Problem Statement

The goal of this project is to build a **robust, leakage-free deep learning pipeline** to predict the **next-day stock price** using historical data.

Instead of directly predicting prices (which are non-stationary), the system predicts **next-day log returns**, and then reconstructs the stock price using inverse transformation. This approach ensures **better statistical properties** and **model stability**.

---

## 2. Core Objectives

- Predict next-day stock prices using **LSTM and GRU**
- Enforce **strict time-series causality**
- Avoid **all forms of data leakage**
- Ensure the pipeline reads **only raw data**
- Fully automate training, evaluation, and reporting
- Produce **reproducible and production-ready code**

---

## 3. High-Level System Design

### Data Flow Overview

```
Raw CSV Files
   │
   ▼
Dataset Loader
   │
   ▼
Feature Engineering (Causal)
   │
   ▼
Sequence Construction
   │
   ▼
LSTM / GRU Training
   │
   ▼
Log-Return Prediction
   │
   ▼
Inverse Price Reconstruction
   │
   ▼
Metrics + Plots
```

All transformations occur **inside the pipeline**.

---

## 4. Dataset Description

### Input Files (`data/raw/`)

| File | Description |
|----|----|
| `StockPrice.csv` | Historical stock prices |
| `Data.csv` | External explanatory signal |

### Important Constraint

> The pipeline **never reads** from processed or interim datasets.  
> Only raw CSV files are accessed.

---

## 5. Target Definition

The learning target is defined as:


<p>
  r<sub>t+1</sub> = log(P<sub>t+1</sub> / P<sub>t</sub>)
</p>

<p>Where:</p>
<ul>
  <li>P<sub>t</sub> = stock price at time <em>t</em></li>
  <li>r<sub>t+1</sub> = next-day log return</li>
</ul>
This formulation:
- Makes the series more stationary
- Improves neural network convergence
- Enables valid inverse transformation

---

## 6. Feature Engineering Strategy

All features are **causal** and use only historical information.

### Feature Categories

- Lagged log returns
- Rolling return statistics (mean, std)
- Volatility windows
- Price trend ratios (shifted)
- Lagged external signals

No same-day or future information is used.

---

## 7. Sequence Construction

- Fixed-length sliding windows (default: 30 days)
- Input shape:
  ```
  (samples, timesteps, features)
  ```
- Output:
  ```
  next-day log return
  ```

This format is compatible with LSTM and GRU architectures.

---

## 8. Models Used

### LSTM (Long Short-Term Memory)
- Captures long-term temporal dependencies
- Handles vanishing gradients

### GRU (Gated Recurrent Unit)
- Simpler architecture
- Faster convergence
- Often performs well on limited data

Both models are trained **without shuffling**.

---

## 9. Training Protocol

- Time-ordered train/test split
- No randomization
- Mean Squared Error loss
- Adam optimizer
- Fixed hyperparameters for reproducibility

---

## 10. Price Reconstruction

Predicted log returns are converted back to prices:

<p>
  P̂<sub>t+1</sub> = P<sub>t</sub> ⋅ e<sup>r̂<sub>t+1</sub></sup>
</p>

This allows intuitive evaluation in price space.

---

## 11. Evaluation Metrics

Metrics are computed on **reconstructed prices**:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

Results are:
- Printed to console
- Saved to `reports/metrics.txt`

---

## 12. Visualization

The pipeline automatically generates:

- Actual price vs LSTM prediction
- Actual price vs GRU prediction

Saved to:
```
reports/figures/lstm_gru_price_predictions.png
```

All plotting logic is centralized in `plots.py`.

---

## 13. Expected Results

Due to market efficiency:

- R² may be low or negative (normal for daily data)
- MAE / RMSE are primary indicators
- Emphasis is on **correctness**, not inflated metrics

---

## 14. How to Run the Project

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Add raw data
```
data/raw/StockPrice.csv
data/raw/Data.csv
```

### Step 3: Run pipeline
```bash
python StockPricePrediction/pipeline.py
```

---

## 15. Design Principles Enforced

- No data leakage
- No future information
- No notebook-only logic
- Pipeline as single source of truth
- Modular, testable code

---

## 16. Future Extensions

- Walk-forward retraining for DL
- Directional classification
- Volatility forecasting
- Trading strategy backtesting
- Hybrid DL + tree models

---

## 17. Conclusion

This project demonstrates a **research-grade, production-ready** approach to time-series modeling for stock prices, prioritizing **methodological correctness** over misleading performance gains.

---

## 18. Author

**Kinjal Mitra**  
