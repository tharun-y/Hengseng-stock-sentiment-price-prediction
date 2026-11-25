
import ast
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

SENTIMENT_FILE = (
    BASE_DIR / "data" / "processed" / "sentiment_results.csv"
)

MARKET_FILE = (
    BASE_DIR / "src" / "collectors" / "data" /
    "raw" / "market" / "hong_kong_market_data.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "next_day_prediction_dataset.csv"

TIMEZONE = "Asia/Hong_Kong"


# These records did not receive valid Groq responses.
# Exclude them rather than treating their default values as real sentiment.
UNCLASSIFIED_IDS = {
    "Telegram-31-1810.HK",
    "NEWS-38-3690.HK",
    "NEWS-39-3690.HK"
}


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def parse_boolean(value):

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true", "1", "yes"
    }


def parse_engagement(value):

    """Convert the saved engagement dictionary into usable values."""

    empty = {
        "views": 0,
        "reactions": 0,
        "likes": 0,
        "replies": 0
    }

    if isinstance(value, dict):
        data = value

    else:
        try:
            data = ast.literal_eval(str(value))
        except (ValueError, SyntaxError, TypeError):
            return empty

    if not isinstance(data, dict):
        return empty

    result = {}

    for key in empty:

        try:
            result[key] = float(data.get(key, 0) or 0)
        except (ValueError, TypeError):
            result[key] = 0

    return result


def to_hong_kong_date(series):

    """Convert UTC timestamps to Hong Kong calendar dates."""

    timestamps = pd.to_datetime(
        series,
        utc=True,
        errors="coerce"
    )

    return (
        timestamps.dt
        .tz_convert(TIMEZONE)
        .dt.strftime("%Y-%m-%d")
    )


# ============================================================
# 3. LOAD MARKET DATA
# ============================================================

def load_market_data():

    print("\nLoading market data...")

    if not MARKET_FILE.exists():
        raise FileNotFoundError(
            f"Market file not found: {MARKET_FILE}"
        )

    df = pd.read_csv(MARKET_FILE)

    required_columns = [
        "timestamp",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing market columns: {missing}"
        )

    # The source timestamps represent daily bars.
    # Convert them to Hong Kong dates before joining sentiment.
    df["date"] = to_hong_kong_date(df["timestamp"])

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=["date", "ticker", "close"]
    )

    df["ticker"] = df["ticker"].astype(str)

    df = df.sort_values(
        ["ticker", "date"]
    )

    # Ensure there is only one market row per ticker/date.
    df = df.drop_duplicates(
        subset=["ticker", "date"],
        keep="last"
    )

    print(f"Market records loaded: {len(df)}")

    return df


# ============================================================
# 4. LOAD AND CLEAN SENTIMENT DATA
# ============================================================

def load_sentiment_data():

    print("\nLoading sentiment data...")

    if not SENTIMENT_FILE.exists():
        raise FileNotFoundError(
            f"Sentiment file not found: {SENTIMENT_FILE}"
        )

    df = pd.read_csv(SENTIMENT_FILE)

    required_columns = [
        "id",
        "ticker",
        "timestamp",
        "source",
        "sentiment",
        "sentiment_score",
        "confidence",
        "relevant",
        "engagement"
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing sentiment columns: {missing}"
        )

    initial_count = len(df)

    # Remove the three records that Groq did not classify.
    df = df[
        ~df["id"].astype(str).isin(UNCLASSIFIED_IDS)
    ].copy()

    print(
        f"Unclassified records excluded: "
        f"{initial_count - len(df)}"
    )

    df["date"] = to_hong_kong_date(df["timestamp"])

    df["relevant"] = df["relevant"].apply(parse_boolean)

    df["sentiment_score"] = pd.to_numeric(
        df["sentiment_score"],
        errors="coerce"
    ).fillna(0)

    df["confidence"] = pd.to_numeric(
        df["confidence"],
        errors="coerce"
    ).fillna(0)

    df["sentiment"] = (
        df["sentiment"]
        .astype(str)
        .str.lower()
    )

    df["source"] = (
        df["source"]
        .astype(str)
        .str.lower()
    )

    df = df.dropna(
        subset=["date", "ticker"]
    )

    # Extract engagement values from the saved dictionaries.
    engagement_values = df["engagement"].apply(
        parse_engagement
    )

    for column in [
        "views",
        "reactions",
        "likes",
        "replies"
    ]:
        df[column] = engagement_values.apply(
            lambda item: item[column]
        )

    print(f"Usable sentiment records: {len(df)}")

    return df


# ============================================================
# 5. AGGREGATE DAILY SENTIMENT
# ============================================================

def aggregate_sentiment(df):

    print("\nAggregating daily sentiment...")

    records = []

    for (ticker, date), group in df.groupby(
        ["ticker", "date"]
    ):

        relevant = group[group["relevant"] == True]

        social = relevant[
            relevant["source"].isin(["telegram", "lihkg"])
        ]

        news = relevant[
            relevant["source"] == "news"
        ]

        records.append({

            "ticker": ticker,
            "date": date,

            "total_mentions": len(group),

            "relevant_mentions": len(relevant),

            "positive_mentions": int(
                (relevant["sentiment"] == "positive").sum()
            ),

            "negative_mentions": int(
                (relevant["sentiment"] == "negative").sum()
            ),

            "neutral_mentions": int(
                (relevant["sentiment"] == "neutral").sum()
            ),

            "average_sentiment": (
                relevant["sentiment_score"].mean()
                if len(relevant) else 0
            ),

            "average_confidence": (
                relevant["confidence"].mean()
                if len(relevant) else 0
            ),

            "social_mentions": len(social),

            "news_mentions": len(news),

            "social_views": social["views"].sum(),

            "social_reactions": social["reactions"].sum(),

            "social_likes": social["likes"].sum(),

            "social_replies": social["replies"].sum()
        })

    sentiment_daily = pd.DataFrame(records)

    print(
        f"Daily ticker sentiment groups: "
        f"{len(sentiment_daily)}"
    )

    return sentiment_daily


# ============================================================
# 6. CALCULATE MARKET FEATURES
# ============================================================

def add_market_features(df):

    print("\nCalculating market indicators...")

    all_groups = []

    for ticker, group in df.groupby("ticker"):

        group = group.sort_values("date").copy()

        close = group["close"]

        # Daily price movement.
        group["daily_return"] = close.pct_change()

        # Moving averages.
        group["ma_5"] = close.rolling(
            window=5,
            min_periods=1
        ).mean()

        group["ma_10"] = close.rolling(
            window=10,
            min_periods=1
        ).mean()

        group["ma_20"] = close.rolling(
            window=20,
            min_periods=1
        ).mean()

        # Price relative to moving averages.
        group["close_vs_ma5"] = (
            close / group["ma_5"] - 1
        )

        group["close_vs_ma20"] = (
            close / group["ma_20"] - 1
        )

        # Historical volatility.
        group["volatility_5"] = (
            group["daily_return"]
            .rolling(5, min_periods=2)
            .std()
        )

        group["volatility_10"] = (
            group["daily_return"]
            .rolling(10, min_periods=2)
            .std()
        )

        # Daily trading range.
        group["intraday_range"] = (
            (group["high"] - group["low"]) / close
        )

        # Volume change.
        group["volume_change"] = (
            group["volume"].pct_change()
        )

        # RSI indicator.
        difference = close.diff()

        gains = difference.clip(lower=0)
        losses = -difference.clip(upper=0)

        average_gain = gains.rolling(
            14,
            min_periods=14
        ).mean()

        average_loss = losses.rolling(
            14,
            min_periods=14
        ).mean()

        relative_strength = (
            average_gain /
            average_loss.replace(0, np.nan)
        )

        group["rsi_14"] = (
            100 - (100 / (1 + relative_strength))
        )

        group["rsi_14"] = group["rsi_14"].fillna(50)

        all_groups.append(group)

    result = pd.concat(
        all_groups,
        ignore_index=True
    )

    result = result.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return result


# ============================================================
# 7. COMBINE SENTIMENT AND MARKET DATA
# ============================================================

def build_dataset():

    market_df = load_market_data()

    sentiment_df = load_sentiment_data()

    sentiment_daily = aggregate_sentiment(
        sentiment_df
    )

    print("\nCombining market and sentiment data...")

    dataset = market_df.merge(
        sentiment_daily,
        on=["ticker", "date"],
        how="left"
    )

    sentiment_columns = [
        "total_mentions",
        "relevant_mentions",
        "positive_mentions",
        "negative_mentions",
        "neutral_mentions",
        "average_sentiment",
        "average_confidence",
        "social_mentions",
        "news_mentions",
        "social_views",
        "social_reactions",
        "social_likes",
        "social_replies"
    ]

    # No mention on a trading day means zero recorded mentions.
    dataset[sentiment_columns] = (
        dataset[sentiment_columns].fillna(0)
    )

    # Add market indicators.
    dataset = add_market_features(dataset)

    dataset = dataset.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    # Target: next available trading day's closing price.
    dataset["next_day_close"] = (
        dataset.groupby("ticker")["close"].shift(-1)
    )

    # Target: next trading day's percentage return.
    dataset["next_day_return"] = (
        dataset.groupby("ticker")["close"].shift(-1)
        / dataset["close"] - 1
    )

    # Remove final row for each ticker because its next-day
    # closing price is not available in the collected history.
    dataset = dataset.dropna(
        subset=["next_day_close", "next_day_return"]
    )

    dataset = dataset.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill indicator gaps caused by the beginning of each history.
    feature_columns = [
        "daily_return",
        "volatility_5",
        "volatility_10",
        "intraday_range",
        "volume_change",
        "close_vs_ma5",
        "close_vs_ma20"
    ]

    dataset[feature_columns] = (
        dataset[feature_columns].fillna(0)
    )

    dataset["volume_change"] = (
        dataset["volume_change"].clip(-5, 5)
    )

    return dataset


# ============================================================
# 8. MAIN
# ============================================================

def main():

    print("\n==========================================")
    print(" HONG KONG NEXT-DAY PREDICTION DATASET")
    print("==========================================")

    dataset = build_dataset()

    dataset.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n==========================================")
    print(" DATASET CREATED SUCCESSFULLY")
    print("==========================================")

    print(f"Total rows: {len(dataset)}")
    print(f"Total columns: {len(dataset.columns)}")

    print("\nRecords per ticker:")

    print(
        dataset.groupby("ticker")
        .size()
        .to_string()
    )

    print("\nDataset columns:")

    print(dataset.columns.tolist())

    print("\nSample records:")

    print(
        dataset[
            [
                "date",
                "ticker",
                "close",
                "average_sentiment",
                "relevant_mentions",
                "next_day_close",
                "next_day_return"
            ]
        ].head(10).to_string(index=False)
    )

    print(f"\nSaved dataset to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
