import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error
)

from config import (
    DATE_COL,
    PRICE_COL,
    TARGET_COL,
    MODEL_DIR,
    REPORTS_DIR,
    SEQUENCE_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE
)

# -----------------------------
# Models
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze()


# -----------------------------
# Helpers
# -----------------------------
def build_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def train_model(model, loader):
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return model


# -----------------------------
# Main entry
# -----------------------------
def train_and_evaluate(df):
    # -----------------------------
    # Separate price & target
    # -----------------------------
    prices = df[PRICE_COL].values
    y = df[TARGET_COL].values

    # -----------------------------
    # DROP NON-NUMERIC COLUMNS
    # -----------------------------
    X = df.drop(columns=[DATE_COL, PRICE_COL, TARGET_COL])

    # -----------------------------
    # Scale features
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    # -----------------------------
    # Build sequences
    # -----------------------------
    X_seq, y_seq = build_sequences(X_scaled, y, SEQUENCE_LENGTH)

    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        ),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # -----------------------------
    # Train models
    # -----------------------------
    lstm = train_model(LSTMModel(X_train.shape[2]), train_loader)
    gru = train_model(GRUModel(X_train.shape[2]), train_loader)

    torch.save(lstm.state_dict(), MODEL_DIR / "best_lstm_model.pt")
    torch.save(gru.state_dict(), MODEL_DIR / "best_gru_model.pt")

    # -----------------------------
    # Predict
    # -----------------------------
    def predict(model):
        model.eval()
        with torch.no_grad():
            return model(
                torch.tensor(X_test, dtype=torch.float32)
            ).numpy()

    lstm_preds = predict(lstm)
    gru_preds = predict(gru)

    # -----------------------------
    # Price reconstruction
    # -----------------------------
    start_idx = len(prices) - len(y_test) - 1
    last_prices = prices[start_idx:start_idx + len(y_test)]

    actual_prices = last_prices * np.exp(y_test)
    lstm_price_preds = last_prices * np.exp(lstm_preds)
    gru_price_preds = last_prices * np.exp(gru_preds)

    # -----------------------------
    # Metrics (EXACT SPEC)
    # -----------------------------
    lstm_mae = mean_absolute_error(actual_prices, lstm_price_preds)
    lstm_rmse = root_mean_squared_error(actual_prices, lstm_price_preds)
    lstm_r2 = r2_score(actual_prices, lstm_price_preds)

    gru_mae = mean_absolute_error(actual_prices, gru_price_preds)
    gru_rmse = root_mean_squared_error(actual_prices, gru_price_preds)
    gru_r2 = r2_score(actual_prices, gru_price_preds)

    metrics = f"""
        LSTM – Log Return Metrics
        MAE  : {lstm_mae:.6f}
        RMSE : {lstm_rmse:.6f}
        R²   : {lstm_r2:.4f}

        GRU – Log Return Metrics
        MAE  : {gru_mae:.6f}
        RMSE : {gru_rmse:.6f}
        R²   : {gru_r2:.4f}
        """
    print(metrics)
    (REPORTS_DIR / "metrics.txt").write_text(metrics)

    return actual_prices, lstm_price_preds, gru_price_preds
