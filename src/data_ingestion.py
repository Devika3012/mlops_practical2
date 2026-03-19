import pandas as pd
from sklearn.datasets import load_diabetes

def load_and_process_data():
    # Load dataset
    data = load_diabetes(as_frame=True)
    df = data.frame

    # Save raw data
    df.to_csv("data/raw/diabetes.csv", index=False)

    # Simple preprocessing (example)
    df_processed = df.copy()

    # Save processed data
    df_processed.to_csv("data/processed/diabetes_processed.csv", index=False)

    print("Data ingestion completed")

if __name__ == "__main__":
    load_and_process_data()