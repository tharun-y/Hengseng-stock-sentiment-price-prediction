
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_FILE = (
    BASE_DIR / "data" / "processed" /
    "next_day_prediction_dataset.csv"
)

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports" / "model_evaluation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "next_day_close"

FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "daily_return",
    "ma_5",
    "ma_10",
    "ma_20",
    "close_vs_ma5",
    "close_vs_ma20",
    "volatility_5",
    "volatility_10",
    "intraday_range",
    "volume_change",
    "rsi_14"
]


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():

    print("\nLoading dataset...")

    if not DATASET_FILE.exists():
        raise FileNotFoundError(DATASET_FILE)

    df = pd.read_csv(DATASET_FILE)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)

    print("Dataset shape:", df.shape)

    return df


# ============================================================
# 3. CHRONOLOGICAL TRAIN-TEST SPLIT
# ============================================================

def split_data(df):

    print("\nSplitting data chronologically...")

    dates = sorted(df["date"].unique())

    split_index = int(len(dates) * 0.80)

    split_date = dates[split_index]

    train = df[df["date"] < split_date].copy()

    test = df[df["date"] >= split_date].copy()

    print("Training cutoff:", split_date)

    print("Training rows:", len(train))

    print("Testing rows:", len(test))

    return train, test


# ============================================================
# 4. EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, model, X_train, y_train, X_test, y_test):

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    mape = (
        np.abs(
            (y_test - predictions) /
            np.maximum(np.abs(y_test), 1e-8)
        ).mean() * 100
    )

    results = {
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")

    return model, predictions, results


# ============================================================
# 5. TRAIN MODELS
# ============================================================

def train_models(df):

    train, test = split_data(df)

    X_train = train[FEATURES]
    X_test = test[FEATURES]

    y_train = train[TARGET]
    y_test = test[TARGET]

    models = {

        "Linear Regression": make_pipeline(
            StandardScaler(),
            LinearRegression()
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )
    }

    all_results = []
    all_predictions = {}

    for name, model in models.items():

        trained_model, predictions, result = evaluate_model(
            name,
            model,
            X_train,
            y_train,
            X_test,
            y_test
        )

        all_results.append(result)

        all_predictions[name] = predictions

        filename = (
            name.lower()
            .replace(" ", "_")
            + ".joblib"
        )

        joblib.dump(
            trained_model,
            MODEL_DIR / filename
        )

    # Baseline: predict next close = current close.
    baseline_predictions = test["close"].to_numpy()

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline_predictions
        )
    )

    baseline_r2 = r2_score(
        y_test,
        baseline_predictions
    )

    all_results.append({
        "model": "Previous Close Baseline",
        "MAE": baseline_mae,
        "RMSE": baseline_rmse,
        "R2": baseline_r2,
        "MAPE": (
            np.abs(
                (y_test.to_numpy() - baseline_predictions) /
                np.maximum(np.abs(y_test.to_numpy()), 1e-8)
            ).mean() * 100
        )
    })

    all_predictions["Previous Close Baseline"] = (
        baseline_predictions
    )

    # Save model comparison.
    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        "MAE"
    ).reset_index(drop=True)

    results_df.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False
    )

    # Save individual test predictions.
    prediction_df = test[
        ["date", "ticker", "close", TARGET]
    ].copy()

    for name, predictions in all_predictions.items():

        column = (
            name.lower()
            .replace(" ", "_")
            + "_prediction"
        )

        prediction_df[column] = predictions

    prediction_df.to_csv(
        REPORT_DIR / "test_predictions.csv",
        index=False
    )

    # Select the best ML model, excluding the baseline.
    ml_results = results_df[
        results_df["model"] != "Previous Close Baseline"
    ]

    best_name = ml_results.iloc[0]["model"]

    best_filename = (
        best_name.lower()
        .replace(" ", "_")
        + ".joblib"
    )

    print("\nBest machine-learning model:", best_name)

    print("\nFinal model comparison:")

    print(results_df.round(4).to_string(index=False))

    print("\nSaved models to:", MODEL_DIR)

    print("Saved evaluation reports to:", REPORT_DIR)

    return results_df


# ============================================================
# 6. MAIN
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("NEXT-DAY STOCK PRICE PREDICTION")
    print("=" * 60)

    df = load_data()

    missing = [
        column for column in FEATURES + [TARGET]
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if df[FEATURES + [TARGET]].isnull().any().any():
        raise ValueError(
            "Missing values detected in model features or target."
        )

    train_models(df)

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
