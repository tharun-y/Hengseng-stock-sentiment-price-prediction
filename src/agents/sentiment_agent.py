"""
Stock Sentiment Analysis Agent Layer
Synthesizes market data, technical indicators, social media discussions,
news sentiment, and ML predictions into a structured analytical summary.
"""

from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
SENTIMENT_RESULTS_FILE = BASE_DIR / "data" / "processed" / "sentiment_results.csv"

COMPANY_NAMES = {
    "0005.HK": "HSBC",
    "0700.HK": "Tencent",
    "9988.HK": "Alibaba",
    "3690.HK": "Meituan",
    "1810.HK": "Xiaomi",
}


def load_raw_sentiments() -> pd.DataFrame:
    """Load granular sentiment items with text and source details."""
    if not SENTIMENT_RESULTS_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(SENTIMENT_RESULTS_FILE)
    if "relevant" in df.columns:
        df = df[df["relevant"] == True].copy()
    return df


def generate_agent_summary(stock_data: Dict[str, Any], raw_sentiments: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Generate a structured analytical interpretation combining:
    1. Recent news
    2. Social media discussions
    3. AI-generated sentiment
    4. Technical indicators
    5. Model predictions
    """
    ticker = stock_data["ticker"]
    company = stock_data.get("company", COMPANY_NAMES.get(ticker, ticker))
    current_price = stock_data["current_price"]
    ml_pred = stock_data["ml_predicted_price"]
    pred_change = stock_data["predicted_change_percent"]
    sentiment_score = stock_data["sentiment_score"]
    sentiment_direction = stock_data["sentiment_direction"]
    pos_mentions = stock_data["positive_mentions"]
    neg_mentions = stock_data["negative_mentions"]
    total_mentions = stock_data["total_mentions"]
    social_score = stock_data.get("social_sentiment_score")
    news_score = stock_data.get("news_sentiment_score")
    social_count = stock_data.get("social_mentions", 0)
    news_count = stock_data.get("news_mentions", 0)
    market_trend = stock_data["market_direction"]
    support = stock_data["support"]
    resistance = stock_data["resistance"]
    rsi = stock_data["rsi"]

    # 1. Analyze signal agreement/disagreement
    ml_direction = "Bullish" if ml_pred > current_price else ("Bearish" if ml_pred < current_price else "Neutral")
    
    agreements = []
    disagreements = []

    # Compare Sentiment vs Technical Trend
    if sentiment_direction in ["Bullish", "Bearish"] and market_trend in ["Bullish", "Bearish"]:
        if sentiment_direction == market_trend:
            agreements.append(f"Social/News sentiment aligns with the {market_trend.lower()} technical trend.")
        else:
            disagreements.append(
                f"Divergence detected: Sentiment is {sentiment_direction.lower()} ({sentiment_score:+.3f}) while technical moving averages show a {market_trend.lower()} trend."
            )

    # Compare ML estimate vs Technical Trend
    if ml_direction != "Neutral" and market_trend != "Sideways":
        if ml_direction != market_trend:
            disagreements.append(
                f"Model divergence: Experimental ML estimate indicates a {pred_change:+.2f}% change ({ml_direction.lower()}), contradicting the prevailing {market_trend.lower()} technical trend."
            )

    # Check RSI condition
    rsi_note = ""
    if rsi < 30:
        rsi_note = f"RSI is currently {rsi:.1f}, signaling oversold territory which may precede a technical rebound or consolidation near support HKD {support:.2f}."
    elif rsi > 70:
        rsi_note = f"RSI is currently {rsi:.1f}, signaling overbought conditions near resistance HKD {resistance:.2f}."
    else:
        rsi_note = f"RSI is at {rsi:.1f}, indicating neutral momentum between support HKD {support:.2f} and resistance HKD {resistance:.2f}."

    # Check Social vs News sentiment divergence
    source_divergence_note = ""
    if social_count > 0 and news_count > 0 and social_score is not None and news_score is not None:
        if (social_score > 0 and news_score < 0) or (social_score < 0 and news_score > 0):
            source_divergence_note = (
                f"Source contrast: Retail social media sentiment ({social_score:+.2f}, {social_count} posts) "
                f"diverges from institutional news coverage ({news_score:+.2f}, {news_count} articles)."
            )
        else:
            source_divergence_note = (
                f"Source consistency: Both social media ({social_score:+.2f}) and news coverage ({news_score:+.2f}) "
                f"lean in the same general direction."
            )
    elif total_mentions == 0:
        source_divergence_note = "Sentiment data warning: No recent relevant social or news mentions were captured for this stock."
    elif social_count == 0:
        source_divergence_note = f"Coverage note: Sentiment is derived solely from institutional news ({news_count} articles); no relevant retail social posts detected."
    elif news_count == 0:
        source_divergence_note = f"Coverage note: Sentiment is derived entirely from social discussions ({social_count} posts); no recent news items detected."

    # Synthesize natural language Agent Interpretation
    interpretation_parts = []

    if total_mentions == 0:
        interpretation_parts.append(
            f"Insufficient sentiment data is available for {company}. Analysis relies predominantly on technical market indicators."
        )
    else:
        sent_phrase = f"{sentiment_direction.lower()} sentiment ({pos_mentions} positive vs {neg_mentions} negative mentions, score: {sentiment_score:+.3f})"
        interpretation_parts.append(f"Public discussions and news show {sent_phrase}.")

    if source_divergence_note:
        interpretation_parts.append(source_divergence_note)

    if market_trend == "Bearish":
        interpretation_parts.append(
            f"Technical indicators confirm a bearish posture, with price trading below the 5-day and 20-day moving averages."
        )
    elif market_trend == "Bullish":
        interpretation_parts.append(
            f"Technical indicators show bullish momentum, trading above short-term moving averages."
        )
    else:
        interpretation_parts.append(
            f"Technical indicators suggest sideways consolidation between support at HKD {support:.2f} and resistance at HKD {resistance:.2f}."
        )

    if rsi_note:
        interpretation_parts.append(rsi_note)

    if disagreements:
        interpretation_parts.append(
            f"Signal divergence: {' '.join(disagreements)} Because experimental ML estimates disagree with technical or sentiment flows, the agent advises caution and avoids a high-conviction directional forecast."
        )
    else:
        interpretation_parts.append(
            f"Signal alignment: Key indicators are broadly coherent. However, since the previous-close baseline statistically matches or exceeds daily ML price forecasts, this experimental projection should not be treated as a guaranteed trading signal."
        )

    agent_interpretation = " ".join(interpretation_parts)

    # Key highlights from real articles/posts
    top_reasons = []
    if raw_sentiments is not None and not raw_sentiments.empty:
        stock_items = raw_sentiments[raw_sentiments["ticker"] == ticker]
        if not stock_items.empty and "reason" in stock_items.columns:
            reasons = stock_items["reason"].dropna().unique().tolist()
            top_reasons = reasons[:3]

    return {
        "ticker": ticker,
        "company": company,
        "market_trend": market_trend,
        "sentiment_direction": sentiment_direction,
        "average_sentiment_score": sentiment_score,
        "positive_mentions": pos_mentions,
        "negative_mentions": neg_mentions,
        "total_mentions": total_mentions,
        "social_mentions": social_count,
        "social_sentiment_score": social_score,
        "news_mentions": news_count,
        "news_sentiment_score": news_score,
        "current_price": current_price,
        "ml_predicted_price": ml_pred,
        "predicted_change_percent": pred_change,
        "support": support,
        "resistance": resistance,
        "rsi": rsi,
        "agreements": agreements,
        "disagreements": disagreements,
        "agent_interpretation": agent_interpretation,
        "top_drivers": top_reasons,
    }
