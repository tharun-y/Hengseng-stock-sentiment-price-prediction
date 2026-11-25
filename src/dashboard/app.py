"""
Interactive Streamlit Dashboard
HengSeng Stock Sentiment & Price Prediction Agent
"""

from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

PREDICTION_FILE = (
    BASE_DIR / "reports" / "predictions" / "latest_stock_predictions.csv"
)

EVAL_FILE = (
    BASE_DIR / "reports" / "model_evaluation" / "overall_comprehensive_evaluation.csv"
)

PER_TICKER_EVAL_FILE = (
    BASE_DIR / "reports" / "model_evaluation" / "per_ticker_return_evaluation.csv"
)

TEST_PRED_FILE = (
    BASE_DIR / "reports" / "model_evaluation" / "test_predictions.csv"
)

st.set_page_config(
    page_title="HengSeng Stock Sentiment & Price Prediction Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #1f77b4;
    }
    .agent-box {
        background-color: #f0f4f8;
        border: 1px solid #d0d7de;
        border-radius: 8px;
        padding: 14px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_all_data():
    df_pred = pd.DataFrame()
    df_eval = pd.DataFrame()
    df_ticker_eval = pd.DataFrame()
    df_test_preds = pd.DataFrame()

    if PREDICTION_FILE.exists():
        df_pred = pd.read_csv(PREDICTION_FILE)
        df_pred["date"] = pd.to_datetime(df_pred["date"])

    if EVAL_FILE.exists():
        df_eval = pd.read_csv(EVAL_FILE)

    if PER_TICKER_EVAL_FILE.exists():
        df_ticker_eval = pd.read_csv(PER_TICKER_EVAL_FILE)

    if TEST_PRED_FILE.exists():
        df_test_preds = pd.read_csv(TEST_PRED_FILE)
        df_test_preds["date"] = pd.to_datetime(df_test_preds["date"])

    return df_pred, df_eval, df_ticker_eval, df_test_preds


df, df_eval, df_ticker_eval, df_test = load_all_data()

# ============================================================
# HEADER
# ============================================================

st.title("📈 HengSeng Stock Sentiment & Price Prediction Agent")
st.markdown(
    "**Multi-Source AI Agent & Quantitative Benchmark Platform for Hong Kong Equities**  \n"
    "*Analyzing: HSBC (0005.HK) • Tencent (0700.HK) • Alibaba (9988.HK) • Meituan (3690.HK) • Xiaomi (1810.HK)*"
)

if df.empty:
    st.error("Prediction file not found. Please run `prediction_engine.py` first.")
    st.stop()


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("🕹️ Dashboard Controls")

view_mode = st.sidebar.radio(
    "Navigation View",
    ["Market Overview & Agent Insights", "Visualizations & Comparative Analytics", "Model Evaluation & Benchmarking", "Data Transparency & Verification"]
)

stock_choices = ["All 5 Stocks"] + [f"{r['company']} ({r['ticker']})" for _, r in df.iterrows()]
selected_stock_label = st.sidebar.selectbox("Filter Stock", stock_choices)

sort_option = st.sidebar.selectbox(
    "Sort Overview Table",
    ["Sentiment Score (High to Low)", "Predicted Change (High to Low)", "Total Mentions (High to Low)", "RSI Momentum"]
)

st.sidebar.divider()
st.sidebar.info(
    "**Agent Status:** Online  \n"
    f"**Latest Trading Date:** {df['date'].max().strftime('%Y-%m-%d')}  \n"
    "**Data Sources:** Apify (LIHKG, Telegram), Exa News, Yahoo Finance, Groq LLM"
)

# Apply stock filter
if selected_stock_label == "All 5 Stocks":
    display_df = df.copy()
else:
    selected_ticker = selected_stock_label.split("(")[-1].replace(")", "").strip()
    display_df = df[df["ticker"] == selected_ticker].copy()

# Apply sorting
if sort_option == "Sentiment Score (High to Low)":
    display_df = display_df.sort_values("sentiment_score", ascending=False)
elif sort_option == "Predicted Change (High to Low)":
    display_df = display_df.sort_values("predicted_change_percent", ascending=False)
elif sort_option == "Total Mentions (High to Low)":
    display_df = display_df.sort_values("total_mentions", ascending=False)
else:
    display_df = display_df.sort_values("rsi", ascending=True)


# ============================================================
# TAB 1: MARKET OVERVIEW & AGENT INSIGHTS
# ============================================================

if view_mode == "Market Overview & Agent Insights":
    st.header("📊 Market Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    total_analyzed = len(display_df)
    bullish_count = int((display_df["sentiment_direction"] == "Bullish").sum())
    bearish_count = int((display_df["sentiment_direction"] == "Bearish").sum())
    neutral_count = int((display_df["sentiment_direction"].isin(["Neutral", "No Data"])).sum())
    total_rel_mentions = int(display_df["relevant_mentions"].sum() if "relevant_mentions" in display_df.columns else display_df["total_mentions"].sum())

    col1.metric("Stocks Analyzed", total_analyzed)
    col2.metric("Bullish Sentiment", bullish_count, help="Positive social/news tone")
    col3.metric("Bearish Sentiment", bearish_count, help="Negative social/news tone")
    col4.metric("Neutral/No Data", neutral_count)
    col5.metric("Total Relevant Mentions", total_rel_mentions)

    st.divider()

    st.header("🤖 Individual Stock Analysis & Agent Interpretation")

    for _, row in display_df.iterrows():
        with st.container(border=True):
            header_col1, header_col2, header_col3 = st.columns([2, 1, 1])
            header_col1.subheader(f"{row['company']} ({row['ticker']})")
            header_col2.caption(f"Trading Date: **{pd.to_datetime(row['date']).strftime('%Y-%m-%d')}**")
            
            # Badge for trend & sentiment
            trend_color = "🔴" if row["market_direction"] == "Bearish" else ("🟢" if row["market_direction"] == "Bullish" else "🟡")
            sent_color = "🟢" if row["sentiment_direction"] == "Bullish" else ("🔴" if row["sentiment_direction"] == "Bearish" else "⚪")
            header_col3.markdown(f"Trend: **{trend_color} {row['market_direction']}** | Sentiment: **{sent_color} {row['sentiment_direction']}**")

            # Metrics Row
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Current Price", f"HK$ {row['current_price']:.2f}")
            m2.metric(
                "ML Forecast (Next-Day)",
                f"HK$ {row['ml_predicted_price']:.2f}",
                f"{row['predicted_change_percent']:+.2f}%",
                delta_color="normal"
            )
            m3.metric("Baseline (Prev Close)", f"HK$ {row['previous_close_baseline']:.2f}", "0.00% (Naive)")
            m4.metric("Sentiment Score", f"{row['sentiment_score']:+.3f}", f"{int(row['positive_mentions'])} pos / {int(row['negative_mentions'])} neg")
            m5.metric("RSI (14-Day)", f"{row['rsi']:.1f}", "Oversold" if row['rsi'] < 30 else ("Overbought" if row['rsi'] > 70 else "Neutral"))

            # Technical Levels & Breakdown
            t_col1, t_col2, t_col3, t_col4 = st.columns(4)
            t_col1.markdown(f"**Support Level:** HK$ {row['support']:.2f}")
            t_col2.markdown(f"**Resistance Level:** HK$ {row['resistance']:.2f}")
            
            social_info = f"{int(row['social_mentions'])} posts"
            if pd.notnull(row.get("social_sentiment_score")):
                social_info += f" (Score: {row['social_sentiment_score']:+.2f})"
            t_col3.markdown(f"**Social Media:** {social_info}")

            news_info = f"{int(row['news_mentions'])} articles"
            if pd.notnull(row.get("news_sentiment_score")):
                news_info += f" (Score: {row['news_sentiment_score']:+.2f})"
            t_col4.markdown(f"**News Coverage:** {news_info}")

            # AI Agent Interpretation Box
            agent_text = row.get("agent_interpretation", "No agent summary generated.")
            disagree_text = row.get("disagreements", "None")

            st.markdown(f"""
            <div class="agent-box">
                <b>🧠 AI Agent Synthesis:</b><br/>
                {agent_text}
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# TAB 2: VISUALIZATIONS & COMPARATIVE ANALYTICS
# ============================================================

elif view_mode == "Visualizations & Comparative Analytics":
    st.header("📈 Visualizations & Comparative Analytics")

    tab_vis1, tab_vis2, tab_vis3 = st.tabs(["Sentiment & Mentions", "Price & Predictions", "Technical Indicators"])

    with tab_vis1:
        st.subheader("Sentiment Score & Volume Breakdown Across Stocks")
        
        c1, c2 = st.columns(2)
        with c1:
            fig_sent = px.bar(
                df,
                x="company",
                y="sentiment_score",
                color="sentiment_direction",
                color_discrete_map={"Bullish": "#2ca02c", "Bearish": "#d62728", "Neutral": "#7f7f7f"},
                title="Average Sentiment Score by Stock",
                labels={"company": "Stock", "sentiment_score": "Sentiment Score (-1 to +1)"},
                text="sentiment_score"
            )
            fig_sent.update_traces(texttemplate="%{text:+.2f}", textposition="outside")
            st.plotly_chart(fig_sent, use_container_width=True)

        with c2:
            mention_df = df.melt(
                id_vars=["company"],
                value_vars=["positive_mentions", "negative_mentions", "neutral_mentions"],
                var_name="Mention Type",
                value_name="Count"
            )
            mention_df["Mention Type"] = mention_df["Mention Type"].replace({
                "positive_mentions": "Positive",
                "negative_mentions": "Negative",
                "neutral_mentions": "Neutral"
            })
            fig_mentions = px.bar(
                mention_df,
                x="company",
                y="Count",
                color="Mention Type",
                barmode="stack",
                color_discrete_map={"Positive": "#2ca02c", "Negative": "#d62728", "Neutral": "#a6cee3"},
                title="Positive vs Negative vs Neutral Mentions",
                labels={"company": "Stock", "Count": "Number of Mentions"}
            )
            st.plotly_chart(fig_mentions, use_container_width=True)

        st.subheader("Social Media vs Institutional News Activity")
        source_df = df.melt(
            id_vars=["company"],
            value_vars=["social_mentions", "news_mentions"],
            var_name="Data Source",
            value_name="Count"
        )
        source_df["Data Source"] = source_df["Data Source"].replace({
            "social_mentions": "Social Media (LIHKG + Telegram)",
            "news_mentions": "News Articles (Exa API)"
        })
        fig_sources = px.bar(
            source_df,
            x="company",
            y="Count",
            color="Data Source",
            barmode="group",
            title="Data Coverage by Source Channel",
            labels={"company": "Stock", "Count": "Item Count"}
        )
        st.plotly_chart(fig_sources, use_container_width=True)

    with tab_vis2:
        st.subheader("Current Price vs Experimental ML Predicted Price")
        
        price_plot_df = df.melt(
            id_vars=["company", "ticker"],
            value_vars=["current_price", "ml_predicted_price", "previous_close_baseline"],
            var_name="Price Type",
            value_name="HKD"
        )
        price_plot_df["Price Type"] = price_plot_df["Price Type"].replace({
            "current_price": "Current Close Price",
            "ml_predicted_price": "ML Forecast Price",
            "previous_close_baseline": "Previous Close Baseline"
        })

        fig_prices = px.bar(
            price_plot_df,
            x="company",
            y="HKD",
            color="Price Type",
            barmode="group",
            title="Comparison of Nominal Stock Price Forecasts",
            labels={"company": "Stock", "HKD": "Price in HKD"}
        )
        st.plotly_chart(fig_prices, use_container_width=True)

        st.subheader("Predicted Percentage Change by Machine Learning Model")
        fig_pct = px.bar(
            df,
            x="company",
            y="predicted_change_percent",
            color="predicted_change_percent",
            color_continuous_scale="RdYlGn",
            title="Expected Next-Day Percentage Change (%)",
            labels={"company": "Stock", "predicted_change_percent": "Predicted Change (%)"},
            text="predicted_change_percent"
        )
        fig_pct.update_traces(texttemplate="%{text:+.2f}%", textposition="outside")
        st.plotly_chart(fig_pct, use_container_width=True)

    with tab_vis3:
        st.subheader("Technical Indicator Overview: RSI, Support & Resistance")
        
        col_rsi, col_sr = st.columns(2)
        with col_rsi:
            fig_rsi = px.bar(
                df,
                x="company",
                y="rsi",
                title="14-Day RSI (Relative Strength Index)",
                labels={"company": "Stock", "rsi": "RSI Value"},
                color="rsi",
                color_continuous_scale="Teal"
            )
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
            st.plotly_chart(fig_rsi, use_container_width=True)

        with col_sr:
            sr_df = df.melt(
                id_vars=["company"],
                value_vars=["support", "current_price", "resistance"],
                var_name="Level",
                value_name="HKD"
            )
            sr_df["Level"] = sr_df["Level"].replace({
                "support": "Support (20-day Low)",
                "current_price": "Current Price",
                "resistance": "Resistance (20-day High)"
            })
            fig_sr = px.scatter(
                sr_df,
                x="company",
                y="HKD",
                color="Level",
                size=[14] * len(sr_df),
                title="Support vs Current Price vs Resistance",
                labels={"company": "Stock", "HKD": "Price (HKD)"}
            )
            st.plotly_chart(fig_sr, use_container_width=True)


# ============================================================
# TAB 3: MODEL EVALUATION & BENCHMARKING
# ============================================================

elif view_mode == "Model Evaluation & Benchmarking":
    st.header("🔬 Machine Learning Model Evaluation & Empirical Benchmarks")

    st.markdown("""
    **Financial Market Reality Check: The Random Walk Hypothesis**  
    In financial forecasting, market prices frequently approximate a random walk or martingale process.
    The table below evaluates four models across the out-of-sample test dataset (chronological 80/20 train-test split, 495 test samples).
    
    *Notice: As expected in quantitative finance, the **Previous Close Baseline** achieves the lowest Price MAE and Return MAE.*
    """)

    if not df_eval.empty:
        st.subheader("Overall Out-of-Sample Performance Comparison")
        st.dataframe(
            df_eval.style.format({
                "Price_MAE": "{:.4f}",
                "Price_RMSE": "{:.4f}",
                "Price_R2": "{:.6f}",
                "Price_MAPE_pct": "{:.2f}%",
                "Return_MAE": "{:.4f}",
                "Return_RMSE": "{:.4f}",
                "Directional_Accuracy": "{:.2%}",
                "Directional_Precision": "{:.4f}",
                "Directional_Recall": "{:.4f}",
                "Directional_F1": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True
        )

    if not df_ticker_eval.empty:
        st.subheader("Per-Stock Model Breakdown")
        st.dataframe(
            df_ticker_eval.style.format({
                "Price_MAE": "{:.4f}",
                "Price_RMSE": "{:.4f}",
                "Return_MAE": "{:.4f}",
                "Directional_Accuracy": "{:.2%}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True
        )

    if not df_test.empty:
        st.subheader("Actual vs Predicted Next-Day Stock Price (Out-of-Sample Test Period)")
        test_stock = st.selectbox("Select Stock for Historical Test Plot", df_test["ticker"].unique())
        sub_test = df_test[df_test["ticker"] == test_stock].sort_values("date")

        fig_test_line = go.Figure()
        fig_test_line.add_trace(go.Scatter(x=sub_test["date"], y=sub_test["next_day_close"], mode="lines", name="Actual Next-Day Close", line=dict(color="#1f77b4", width=2)))
        fig_test_line.add_trace(go.Scatter(x=sub_test["date"], y=sub_test["previous_close_baseline_prediction"], mode="lines", name="Previous Close Baseline", line=dict(color="gray", dash="dot")))
        fig_test_line.add_trace(go.Scatter(x=sub_test["date"], y=sub_test["linear_regression_prediction"], mode="lines", name="Linear Regression", line=dict(color="orange", width=1.5)))
        fig_test_line.add_trace(go.Scatter(x=sub_test["date"], y=sub_test["xgboost_prediction"], mode="lines", name="XGBoost", line=dict(color="green", width=1.5)))

        fig_test_line.update_layout(title=f"{test_stock} - Actual vs Predicted Test Trajectory", xaxis_title="Date", yaxis_title="Price (HKD)", height=450)
        st.plotly_chart(fig_test_line, use_container_width=True)


# ============================================================
# TAB 4: DATA TRANSPARENCY & VERIFICATION
# ============================================================

elif view_mode == "Data Transparency & Verification":
    st.header("🔍 Data Transparency, Sources & Methodology")

    st.markdown("""
    ### Data Pipeline Architecture
    1. **Market Data**: Real daily OHLCV prices collected via Yahoo Finance (`yfinance`) covering daily bars with 5/10/20-day moving averages, RSI(14), and volatility metrics.
    2. **Social Media Discussions**: Scraped via **Apify** actors from:
       - **LIHKG**: Top Hong Kong retail investor forum (`claude_code_reviewer/lihkg-scraper-en`)
       - **Telegram**: Financial channels (`k-data-lab/telegram-channel-scraper`)
    3. **Institutional News**: Live search using **Exa API** targeting recent financial and market news articles.
    4. **AI Sentiment Extraction**: Structured multi-language sentiment classification via **Groq LLM** (`openai/gpt-oss-20b`).
    """)

    st.subheader("Dataset Preview: Current Predictions")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    st.warning("""
    **Mandatory Educational & Experimental Disclaimer**  
    This dashboard and its underlying machine learning models are designed strictly for academic, research, and technical demonstration purposes.
    Financial markets involve substantial risk. Past model performance is not indicative of future returns, and experimental predictions must never be construed as financial advice.
    """)
