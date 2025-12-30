from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# -----------------------------
# Directories
# -----------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# -----------------------------
# Columns
# -----------------------------
DATE_COL = "Date"
PRICE_COL = "Price"
EXTERNAL_COL = "Data"

TARGET_COL = "target_log_return"

# -----------------------------
# DL Hyperparameters
# -----------------------------
SEQUENCE_LENGTH = 30
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
