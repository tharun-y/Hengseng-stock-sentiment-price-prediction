
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "next_day_prediction_dataset.csv"
)

OUTPUT_DIR = BASE_DIR / "reports" / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

def load_data():

    print("\nLoading prediction dataset...")

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}"
        )

    df = pd.read_csv(DATASET_FILE)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["ticker", "date"]
    )

    print(f"Dataset loaded: {df.shape[0]} rows")
    print(f"Features available: {df.shape[1]} columns")

    return df


# ============================================================
# 3. BASIC DATA ANALYSIS
# ============================================================

def basic_analysis(df):

    print("\n" + "=" * 60)
    print("BASIC DATA ANALYSIS")
    print("=" * 60)

    print("\nDataset shape:")
    print(df.shape)

    print("\nStocks available:")
    print(df["ticker"].value_counts())

    print("\nDate range:")
    print(df["date"].min(), "to", df["date"].max())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nStatistical summary:")
    print(df.describe().round(3).to_string())

    print("\nSentiment coverage:")

    coverage = df.groupby("ticker").agg(
        trading_days=("date", "count"),
        days_with_mentions=(
            "relevant_mentions",
            lambda x: (x > 0).sum()
        ),
        total_mentions=("relevant_mentions", "sum"),
        average_sentiment=("average_sentiment", "mean")
    )

    print(coverage.round(3).to_string())

    coverage.to_csv(
        OUTPUT_DIR / "sentiment_coverage.csv"
    )

    return coverage


# ============================================================
# 4. STOCK PRICE MOVEMENT
# ============================================================

def plot_stock_prices(df):

    print("\nGenerating stock price charts...")

    fig = px.line(
        df,
        x="date",
        y="close",
        color="ticker",
        title="Historical Closing Prices - Five Hong Kong Stocks",
        labels={
            "date": "Trading Date",
            "close": "Closing Price",
            "ticker": "Stock"
        }
    )

    fig.update_layout(
        hovermode="x unified",
        template="plotly_white"
    )

    fig.write_html(
        OUTPUT_DIR / "stock_price_trends.html"
    )


# ============================================================
# 5. SENTIMENT TRENDS
# ============================================================

def plot_sentiment_trends(df):

    print("\nGenerating sentiment trend charts...")

    sentiment_df = df[
        df["relevant_mentions"] > 0
    ].copy()

    if sentiment_df.empty:
        print("No relevant sentiment data available.")
        return

    fig = px.line(
        sentiment_df,
        x="date",
        y="average_sentiment",
        color="ticker",
        markers=True,
        title="Daily Average Sentiment by Stock",
        labels={
            "date": "Trading Date",
            "average_sentiment": "Average Sentiment Score",
            "ticker": "Stock"
        }
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        annotation_text="Neutral"
    )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified"
    )

    fig.write_html(
        OUTPUT_DIR / "sentiment_trends.html"
    )


# ============================================================
# 6. POSITIVE VS NEGATIVE MENTIONS
# ============================================================

def plot_sentiment_distribution(df):

    print("\nGenerating sentiment distribution chart...")

    summary = df.groupby("ticker").agg(
        positive=("positive_mentions", "sum"),
        negative=("negative_mentions", "sum"),
        neutral=("neutral_mentions", "sum")
    ).reset_index()

    melted = summary.melt(
        id_vars="ticker",
        var_name="sentiment",
        value_name="mentions"
    )

    fig = px.bar(
        melted,
        x="ticker",
        y="mentions",
        color="sentiment",
        barmode="group",
        title="Positive, Negative and Neutral Mentions",
        labels={
            "ticker": "Stock",
            "mentions": "Number of Mentions",
            "sentiment": "Sentiment"
        }
    )

    fig.update_layout(
        template="plotly_white"
    )

    fig.write_html(
        OUTPUT_DIR / "sentiment_distribution.html"
    )


# ============================================================
# 7. SENTIMENT VS NEXT-DAY RETURN
# ============================================================

def plot_sentiment_vs_returns(df):

    print("\nGenerating sentiment vs return chart...")

    sentiment_df = df[
        df["relevant_mentions"] > 0
    ].copy()

    if sentiment_df.empty:
        print("No sentiment observations available.")
        return

    fig = px.scatter(
        sentiment_df,
        x="average_sentiment",
        y="next_day_return",
        color="ticker",
        size="relevant_mentions",
        hover_data=[
            "date",
            "close",
            "next_day_close",
            "relevant_mentions"
        ],
        title="Sentiment vs Next Trading Day Return",
        labels={
            "average_sentiment": "Average Sentiment Score",
            "next_day_return": "Next-Day Return",
            "ticker": "Stock",
            "relevant_mentions": "Relevant Mentions"
        }
    )

    fig.add_hline(
        y=0,
        line_dash="dash"
    )

    fig.add_vline(
        x=0,
        line_dash="dash"
    )

    fig.update_layout(
        template="plotly_white"
    )

    fig.write_html(
        OUTPUT_DIR / "sentiment_vs_returns.html"
    )


# ============================================================
# 8. SOCIAL AND NEWS ACTIVITY
# ============================================================

def plot_information_activity(df):

    print("\nGenerating social and news activity chart...")

    activity = df.groupby("ticker").agg(
        social_mentions=("social_mentions", "sum"),
        news_mentions=("news_mentions", "sum")
    ).reset_index()

    melted = activity.melt(
        id_vars="ticker",
        var_name="source",
        value_name="mentions"
    )

    fig = px.bar(
        melted,
        x="ticker",
        y="mentions",
        color="source",
        barmode="group",
        title="Social Media vs News Activity",
        labels={
            "ticker": "Stock",
            "mentions": "Number of Mentions",
            "source": "Information Source"
        }
    )

    fig.update_layout(
        template="plotly_white"
    )

    fig.write_html(
        OUTPUT_DIR / "social_vs_news_activity.html"
    )


# ============================================================
# 9. FEATURE CORRELATION
# ============================================================

def plot_feature_correlation(df):

    print("\nGenerating feature correlation heatmap...")

    columns = [
        "close",
        "volume",
        "daily_return",
        "ma_5",
        "ma_20",
        "rsi_14",
        "volatility_5",
        "average_sentiment",
        "relevant_mentions",
        "positive_mentions",
        "negative_mentions",
        "next_day_return"
    ]

    correlation = df[columns].corr(
        numeric_only=True
    )

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Feature Correlation Heatmap"
    )

    fig.update_layout(
        template="plotly_white",
        width=1100,
        height=900
    )

    fig.write_html(
        OUTPUT_DIR / "feature_correlation.html"
    )

    correlation.to_csv(
        OUTPUT_DIR / "feature_correlation.csv"
    )


# ============================================================
# 10. GENERATE EDA REPORT
# ============================================================

def generate_report(df, coverage):

    print("\nGenerating EDA summary report...")

    report = f"""
    <html>
    <head>
        <title>Hong Kong Stock Sentiment EDA</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background: #f7f7f7;
            }}
            h1, h2 {{
                color: #222;
            }}
            a {{
                display: block;
                margin: 12px 0;
                color: #1565c0;
            }}
            .section {{
                background: white;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 8px;
            }}
        </style>
    </head>
    <body>

        <h1>Hong Kong Stock Sentiment Analysis</h1>

        <div class="section">
            <h2>Dataset Overview</h2>
            <p>Total records: {len(df)}</p>
            <p>Total columns: {len(df.columns)}</p>
            <p>Stocks analysed: {df['ticker'].nunique()}</p>
            <p>Start date: {df['date'].min().date()}</p>
            <p>End date: {df['date'].max().date()}</p>
        </div>

        <div class="section">
            <h2>Visualizations</h2>

            <a href="stock_price_trends.html">
                Stock Price Trends
            </a>

            <a href="sentiment_trends.html">
                Sentiment Trends
            </a>

            <a href="sentiment_distribution.html">
                Sentiment Distribution
            </a>

            <a href="sentiment_vs_returns.html">
                Sentiment vs Next-Day Returns
            </a>

            <a href="social_vs_news_activity.html">
                Social Media vs News Activity
            </a>

            <a href="feature_correlation.html">
                Feature Correlation Heatmap
            </a>
        </div>

        <div class="section">
            <h2>Sentiment Coverage</h2>
            {coverage.to_html()}
        </div>

    </body>
    </html>
    """

    report_file = OUTPUT_DIR / "index.html"

    report_file.write_text(
        report,
        encoding="utf-8"
    )

    print(f"EDA report saved: {report_file}")


# ============================================================
# 11. MAIN
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("HONG KONG STOCK SENTIMENT - EDA")
    print("=" * 60)

    df = load_data()

    coverage = basic_analysis(df)

    plot_stock_prices(df)

    plot_sentiment_trends(df)

    plot_sentiment_distribution(df)

    plot_sentiment_vs_returns(df)

    plot_information_activity(df)

    plot_feature_correlation(df)

    generate_report(df, coverage)

    print("\n" + "=" * 60)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nAll reports saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
