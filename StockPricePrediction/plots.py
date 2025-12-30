# StockPricePrediction/plots.py

import matplotlib.pyplot as plt
from pathlib import Path

from config import FIGURES_DIR


def plot_price_predictions(
    actual_prices,
    lstm_price_preds,
    gru_price_preds,
    filename="lstm_gru_price_predictions.png"
):
    """
    Plot actual vs LSTM vs GRU predicted prices
    and save the figure to reports/figures/.
    """

    FIGURES_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(actual_prices, label="Actual Price", linewidth=2)
    plt.plot(lstm_price_preds, label="LSTM Predicted", alpha=0.7)
    plt.plot(gru_price_preds, label="GRU Predicted", alpha=0.7)
    plt.legend()
    plt.title("Next-Day Stock Price Prediction via Log Returns")
    plt.tight_layout()

    output_path = FIGURES_DIR / filename
    plt.savefig(output_path)
    plt.close()

    print(f"Price prediction plot saved to: {output_path}")
