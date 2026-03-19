import pandas as pd
import os

def load_and_process_data():
    df = pd.read_csv("data/raw/diabetes.csv")

    os.makedirs("data/processed", exist_ok=True)

    df.to_csv("data/processed/diabetes_processed.csv", index=False)

    print("Data ingestion complete")

if __name__ == "__main__":
    load_and_process_data()