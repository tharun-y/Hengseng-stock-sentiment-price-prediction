
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parents[2]

FILE = (
    BASE_DIR / "reports" / "model_evaluation" /
    "test_predictions.csv"
)

OUTPUT = (
    BASE_DIR / "reports" / "model_evaluation" /
    "per_ticker_comparison.csv"
)


def main():

    df = pd.read_csv(FILE)

    models = {
        "Previous Close Baseline": "previous_close_baseline_prediction",
        "Linear Regression": "linear_regression_prediction",
        "Random Forest": "random_forest_prediction",
        "XGBoost": "xgboost_prediction"
    }

    results = []

    for ticker, group in df.groupby("ticker"):

        actual = group["next_day_close"]

        for model_name, column in models.items():

            predicted = group[column]

            results.append({
                "ticker": ticker,
                "model": model_name,
                "MAE": mean_absolute_error(actual, predicted),
                "RMSE": np.sqrt(
                    mean_squared_error(actual, predicted)
                )
            })

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        ["ticker", "MAE"]
    )

    result_df.to_csv(OUTPUT, index=False)

    print("\nPER-STOCK MODEL PERFORMANCE")
    print("=" * 65)

    print(
        result_df.round(4).to_string(index=False)
    )

    print("\nBest model for each stock:")

    best = (
        result_df.sort_values("MAE")
        .groupby("ticker")
        .first()
        .reset_index()
    )

    print(
        best[["ticker", "model", "MAE", "RMSE"]]
        .round(4)
        .to_string(index=False)
    )

    print(f"\nSaved report: {OUTPUT}")


if __name__ == "__main__":
    main()
