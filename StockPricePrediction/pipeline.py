from dataset import load_raw_data
from features import create_features
from modeling.train import train_and_evaluate
from plots import plot_price_predictions


def main():
    df = load_raw_data()
    df = create_features(df)

    actual_prices, lstm_preds, gru_preds = train_and_evaluate(df)

    plot_price_predictions(
        actual_prices,
        lstm_preds,
        gru_preds
    )


if __name__ == "__main__":
    main()
