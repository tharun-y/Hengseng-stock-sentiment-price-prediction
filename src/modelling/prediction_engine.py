"""
Prediction Engine for Hong Kong Equities
Integrates historical market data, technical indicators, social and news sentiment,
and trained ML models to generate multi-dimensional predictions and agent insights.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
import numpy as np
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.agents.sentiment_agent import generate_agent_summary, load_raw_sentiments

# ============================================================
# CONFIGURATION & ASSET METADATA
# ============================================================

DATASET_FILE = BASE_DIR / "data" / "processed" / "next_day_prediction_dataset.csv"
SENTIMENT_RESULTS_FILE = BASE_DIR / "data" / "processed" / "sentiment_results.csv"
SENTIMENT_SUMMARY_FILE = BASE_DIR / "data" / "processed" / "sentiment_summary.csv"
MODEL_FILE = BASE_DIR / "models" / "linear_regression.joblib"
OUTPUT_DIR = BASE_DIR / "reports" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = {
    "0005.HK": "HSBC",
    "0700.HK": "Tencent",
    "9988.HK": "Alibaba",
    "3690.HK": "Meituan",
    "1810.HK": "Xiaomi",
}

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
    "rsi_14",
]


# ============================================================
# DATA LOADING & PREPARATION
# ============================================================

def load_data():
    """Load market dataset, sentiment files, and trained model."""
    print("\nLoading market data and sentiment records...")

    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Market dataset missing: {DATASET_FILE}")
    market = pd.read_csv(DATASET_FILE)
    market["date"] = pd.to_datetime(market["date"])
    market = market.sort_values(["ticker", "date"])

    sentiment_results = pd.DataFrame()
    if SENTIMENT_RESULTS_FILE.exists():
        sentiment_results = pd.read_csv(SENTIMENT_RESULTS_FILE)

    sentiment_summary = pd.DataFrame()
    if SENTIMENT_SUMMARY_FILE.exists():
        sentiment_summary = pd.read_csv(SENTIMENT_SUMMARY_FILE)

    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file missing: {MODEL_FILE}")
    model = joblib.load(MODEL_FILE)

    return market, sentiment_summary, sentiment_results, model


# ============================================================
# MULTI-SOURCE SENTIMENT BREAKDOWN
# ============================================================

def extract_sentiment_profile(ticker: str, summary_df: pd.DataFrame, results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract overall, social media, and news sentiment breakdown for a stock.
    Distinguishes clearly when data is unavailable or insufficient.
    """
    profile = {
        "sentiment_score": 0.0,
        "sentiment_direction": "No Data",
        "has_sentiment_data": False,
        "positive_mentions": 0,
        "negative_mentions": 0,
        "neutral_mentions": 0,
        "total_mentions": 0,
        "relevant_mentions": 0,
        "social_mentions": 0,
        "social_sentiment_score": None,
        "news_mentions": 0,
        "news_sentiment_score": None,
    }

    # Extract from summary if available
    if not summary_df.empty and ticker in summary_df["ticker"].values:
        row = summary_df[summary_df["ticker"] == ticker].iloc[0]
        pos = int(row.get("positive_mentions", 0) or 0)
        neg = int(row.get("negative_mentions", 0) or 0)
        neu = int(row.get("neutral_mentions", 0) or 0)
        tot = int(row.get("total_mentions", 0) or 0)
        rel = int(row.get("relevant_mentions", 0) or 0)
        avg_score = float(row.get("average_sentiment_score", 0.0) or 0.0)

        profile["positive_mentions"] = pos
        profile["negative_mentions"] = neg
        profile["neutral_mentions"] = neu
        profile["total_mentions"] = tot
        profile["relevant_mentions"] = rel
        profile["sentiment_score"] = round(avg_score, 4)

        if rel > 0:
            profile["has_sentiment_data"] = True
            if pos > neg and avg_score > 0.05:
                profile["sentiment_direction"] = "Bullish"
            elif neg > pos and avg_score < -0.05:
                profile["sentiment_direction"] = "Bearish"
            else:
                profile["sentiment_direction"] = "Neutral"

    # Extract source-level details (Social vs News) from raw results
    if not results_df.empty and "ticker" in results_df.columns:
        stock_res = results_df[results_df["ticker"] == ticker]
        if "relevant" in stock_res.columns:
            stock_rel = stock_res[stock_res["relevant"] == True]
        else:
            stock_rel = stock_res

        social_mask = stock_rel["source"].isin(["LIHKG", "Telegram"])
        news_mask = stock_rel["source"] == "News"

        social_df = stock_rel[social_mask]
        news_df = stock_rel[news_mask]

        profile["social_mentions"] = len(social_df)
        if len(social_df) > 0 and "sentiment_score" in social_df.columns:
            profile["social_sentiment_score"] = round(float(social_df["sentiment_score"].mean()), 4)

        profile["news_mentions"] = len(news_df)
        if len(news_df) > 0 and "sentiment_score" in news_df.columns:
            profile["news_sentiment_score"] = round(float(news_df["sentiment_score"].mean()), 4)

    return profile


# ============================================================
# TECHNICAL ANALYSIS ENGINE
# ============================================================

def calculate_technical_levels(history: pd.DataFrame) -> Tuple[float, float, str]:
    """
    Compute explainable support and resistance levels from rolling 20-day
    highs and lows, and determine trend from short vs medium moving averages.
    """
    recent = history.tail(20)
    support = float(recent["low"].min())
    resistance = float(recent["high"].max())

    latest = history.iloc[-1]
    close = float(latest["close"])
    ma5 = float(latest["ma_5"])
    ma20 = float(latest["ma_20"])

    if close > ma5 and ma5 > ma20:
        trend = "Bullish"
    elif close < ma5 and ma5 < ma20:
        trend = "Bearish"
    else:
        trend = "Sideways"

    return round(support, 2), round(resistance, 2), trend


# ============================================================
# PREDICTION GENERATION
# ============================================================

def generate_all_stock_predictions():
    """Run full inference across all 5 Hong Kong stocks and synthesize agent summaries."""
    market, sentiment_summary, sentiment_results, model = load_data()
    raw_rel_sentiments = load_raw_sentiments()

    predictions = []

    for ticker, company in ASSETS.items():
        stock_history = market[market["ticker"] == ticker].sort_values("date")
        if stock_history.empty:
            print(f"Warning: No market history for {ticker}")
            continue

        latest = stock_history.iloc[-1]
        latest_date = latest["date"]
        current_price = round(float(latest["close"]), 2)
        baseline_pred = current_price

        # ML next-day price inference
        X = latest[FEATURES].to_frame().T
        ml_pred = round(float(model.predict(X)[0]), 2)
        pct_change = round(((ml_pred / current_price) - 1) * 100, 3)

        support, resistance, market_trend = calculate_technical_levels(stock_history)
        rsi = round(float(latest["rsi_14"]), 2)

        sentiment_info = extract_sentiment_profile(ticker, sentiment_summary, sentiment_results)

        stock_record = {
            "ticker": ticker,
            "company": company,
            "date": latest_date.strftime("%Y-%m-%d"),
            "current_price": current_price,
            "ml_predicted_price": ml_pred,
            "previous_close_baseline": baseline_pred,
            "predicted_change_percent": pct_change,
            "market_direction": market_trend,
            "sentiment_score": sentiment_info["sentiment_score"],
            "sentiment_direction": sentiment_info["sentiment_direction"],
            "has_sentiment_data": sentiment_info["has_sentiment_data"],
            "positive_mentions": sentiment_info["positive_mentions"],
            "negative_mentions": sentiment_info["negative_mentions"],
            "neutral_mentions": sentiment_info["neutral_mentions"],
            "total_mentions": sentiment_info["total_mentions"],
            "relevant_mentions": sentiment_info["relevant_mentions"],
            "social_mentions": sentiment_info["social_mentions"],
            "social_sentiment_score": sentiment_info["social_sentiment_score"],
            "news_mentions": sentiment_info["news_mentions"],
            "news_sentiment_score": sentiment_info["news_sentiment_score"],
            "support": support,
            "resistance": resistance,
            "rsi": rsi,
        }

        # Generate agent structured synthesis
        agent_output = generate_agent_summary(stock_record, raw_rel_sentiments)
        stock_record["agent_interpretation"] = agent_output["agent_interpretation"]
        stock_record["disagreements"] = "; ".join(agent_output["disagreements"]) if agent_output["disagreements"] else "None (signals coherent)"

        predictions.append(stock_record)

    pred_df = pd.DataFrame(predictions)
    pred_df = pred_df.sort_values("sentiment_score", ascending=False).reset_index(drop=True)

    # Save to reports/predictions
    out_csv = OUTPUT_DIR / "latest_stock_predictions.csv"
    pred_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 90)
    print("ALL 5 HONG KONG STOCKS PREDICTION & AGENT ANALYSIS COMPLETED")
    print("=" * 90)
    display_cols = [
        "company", "ticker", "current_price", "ml_predicted_price",
        "predicted_change_percent", "market_direction", "sentiment_direction",
        "sentiment_score", "social_mentions", "news_mentions", "rsi"
    ]
    print(pred_df[display_cols].to_string(index=False))

    print("\nSample Agent Synthesis:")
    for _, row in pred_df.iterrows():
        print(f"\n[{row['company']} - {row['ticker']}]")
        print(f"Agent Interpretation: {row['agent_interpretation']}")

    print(f"\nSaved predictions to: {out_csv}")
    return pred_df


if __name__ == "__main__":
    generate_all_stock_predictions()
