"""
Train a Linear Regression model for real estate price prediction and save it to disk.

Usage:
    python train_model.py --csv /path/to/Real_Estate.csv --out real_estate_model.pkl
"""
import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="Real_Estate.csv",
                        help="Path to Real_Estate.csv")
    parser.add_argument("--out", type=str, default="real_estate_model.pkl",
                        help="Where to save the trained model")
    args = parser.parse_args()

    # Load the dataset
    df = pd.read_csv(args.csv)

    # Define features and target (must match CSV column names exactly)
    features = [
        'Distance to the nearest MRT station',
        'Number of convenience stores',
        'Latitude',
        'Longitude'
    ]
    target = 'House price of unit area'

    X = df[features]
    y = df[target]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("✅ Training complete")
    print(f"Samples: train={len(X_train)}, test={len(X_test)}")
    print(f"RMSE: {rmse:.3f} | R^2: {r2:.3f}")

    # Save model + feature names
    artifact = {
        "model": model,
        "feature_names": features
    }
    joblib.dump(artifact, args.out)
    print(f"💾 Saved model artifact to: {args.out}")

if __name__ == "__main__":
    main()
