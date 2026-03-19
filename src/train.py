import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def train_model():
    df = pd.read_csv("data/processed/diabetes_processed.csv")

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, "models/model.pkl")

    print("Model trained and saved")

if __name__ == "__main__":
    train_model()