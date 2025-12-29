# run_pipeline.py

from dataset import load_and_merge_raw_data
from features import create_features
from data_processing import save_processed_data
from modeling.train import train_xgboost

def main():
    # Load + merge raw data
    df = load_and_merge_raw_data()

    # Feature engineering
    df = create_features(df)

    # Save processed dataset
    save_processed_data(df)

    # Train model
    train_xgboost(df)

if __name__ == "__main__":
    main()
