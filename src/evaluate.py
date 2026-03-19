import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

def evaluate_model():
    df = pd.read_csv("data/processed/diabetes_processed.csv")

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    model = joblib.load("models/model.pkl")

    predictions = model.predict(X)

    mse = mean_squared_error(y, predictions)

    print("MSE:", mse)

if __name__ == "__main__":
    evaluate_model()