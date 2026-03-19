import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
def train_model():
    # Load processed data
    df = pd.read_csv("data/processed/diabetes_processed.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    # Save model
    joblib.dump(model, "models/linear_model.pkl")
    print("Model training completed")
if __name__ == "__main__":
    train_model()