# Model Evaluation & Benchmark Report

## 1. Executive Summary

This report evaluates next-day price and return predictions across 5 Hong Kong benchmark equities:
**HSBC (0005.HK), Tencent (0700.HK), Alibaba (9988.HK), Meituan (3690.HK), and Xiaomi (1810.HK)**.

Evaluations were performed using a strict **chronological train/test split (80% train, 20% test)** on daily OHLCV and technical feature data (July 2026 to November 2026 test window) to prevent look-ahead bias and data leakage.

### Benchmark Finding: The Random Walk Hypothesis in Financial Markets
In line with standard quantitative finance literature (Malkiel, 1973; Fama, 1970), the **Previous-Close Baseline (Naive persistence forecast: $\hat{y}_{t+1} = y_t$)** outperforms all machine learning models in next-day price MAE and RMSE:

- **Previous Close Baseline**: Price MAE = **2.7623**, RMSE = **4.8777**, Return MAE = **0.0190**
- **Linear Regression**: Price MAE = **2.8622**, RMSE = **4.9656**, Return MAE = **0.0206**
- **XGBoost**: Price MAE = **3.1730**, RMSE = **5.4524**, Return MAE = **0.0231**
- **Random Forest**: Price MAE = **3.5355**, RMSE = **6.1923**, Return MAE = **0.0242**

**Key Insight:** Machine learning models that predict nominal price levels in financial time series are prone to slight lag/overfitting on noise. The previous-close baseline predicts zero change (delta y = 0), which minimizes expected squared error under a martingale/near-random-walk process. We explicitly report this behavior without fabricating or claiming ML superiority.

---

## 2. Overall Performance Metrics

| Model | Price MAE | Price RMSE | Return MAE | Return RMSE | Directional Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|---|---|
| Previous Close Baseline | 2.7623 | 4.8777 | 0.0190 | 0.0260 | 0.5495 | 0.0000 | 0.0000 | 0.0000 |
| Linear Regression | 2.8622 | 4.9656 | 0.0206 | 0.0281 | 0.4626 | 0.4266 | 0.5605 | 0.4845 |
| XGBoost | 3.1730 | 5.4524 | 0.0231 | 0.0310 | 0.5131 | 0.4605 | 0.4709 | 0.4656 |
| Random Forest | 3.5355 | 6.1923 | 0.0242 | 0.0328 | 0.4970 | 0.4375 | 0.4081 | 0.4223 |

> **Note on Baseline Directional Accuracy:**
> The previous close baseline predicts exactly zero price change every day. In binary Up/Down classification (where Up is defined as strictly positive return r > 0 and Down includes non-positive returns r <= 0), the naive baseline default predicts "not up" for all instances. Because the test set had a slight downward bias (54.95% down or flat trading days), assigning all instances to the non-up class mechanically yields a 54.95% accuracy with 0.0000 precision and recall. Conversely, XGBoost achieves 51.31% directional accuracy with a balanced F1 of 0.4656, and Linear Regression achieves 46.26% with an F1 of 0.4845.

---

## 3. Per-Stock Evaluation Breakdown

| Ticker | Model | Price MAE | Price RMSE | Return MAE | Directional Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|---|---|
| 0005.HK | Previous Close Baseline | 1.8071 | 2.4432 | 0.0119 | 0.5354 | 0.0000 | 0.0000 | 0.0000 |
| 0005.HK | Linear Regression | 1.8625 | 2.4638 | 0.0123 | 0.4747 | 0.4400 | 0.4783 | 0.4583 |
| 0005.HK | XGBoost | 1.9408 | 2.4796 | 0.0128 | 0.5657 | 0.5600 | 0.3043 | 0.3944 |
| 0005.HK | Random Forest | 2.2003 | 2.7629 | 0.0144 | 0.5556 | 0.5500 | 0.2391 | 0.3333 |
| 0700.HK | Previous Close Baseline | 7.4667 | 9.9029 | 0.0172 | 0.5859 | 0.0000 | 0.0000 | 0.0000 |
| 0700.HK | Linear Regression | 7.5779 | 10.0456 | 0.0175 | 0.4141 | 0.3509 | 0.4878 | 0.4082 |
| 0700.HK | XGBoost | 8.6654 | 11.1686 | 0.0203 | 0.5556 | 0.4651 | 0.4878 | 0.4762 |
| 0700.HK | Random Forest | 10.1624 | 12.8659 | 0.0241 | 0.4242 | 0.3261 | 0.3659 | 0.3448 |
| 1810.HK | Previous Close Baseline | 0.6745 | 0.8822 | 0.0249 | 0.5556 | 0.0000 | 0.0000 | 0.0000 |
| 1810.HK | Linear Regression | 0.8073 | 1.0339 | 0.0298 | 0.4444 | 0.4127 | 0.5909 | 0.4860 |
| 1810.HK | XGBoost | 1.0039 | 1.2353 | 0.0370 | 0.4444 | 0.3830 | 0.4091 | 0.3956 |
| 1810.HK | Random Forest | 0.9550 | 1.2489 | 0.0356 | 0.5152 | 0.4167 | 0.2273 | 0.2941 |
| 3690.HK | Previous Close Baseline | 1.5061 | 1.9785 | 0.0193 | 0.5556 | 0.0000 | 0.0000 | 0.0000 |
| 3690.HK | Linear Regression | 1.6168 | 2.1164 | 0.0208 | 0.4646 | 0.4328 | 0.6591 | 0.5225 |
| 3690.HK | XGBoost | 1.7105 | 2.1544 | 0.0221 | 0.5253 | 0.4754 | 0.6591 | 0.5524 |
| 3690.HK | Random Forest | 1.7430 | 2.2268 | 0.0225 | 0.4343 | 0.3929 | 0.5000 | 0.4400 |
| 9988.HK | Previous Close Baseline | 2.3571 | 3.1983 | 0.0218 | 0.5152 | 0.0000 | 0.0000 | 0.0000 |
| 9988.HK | Linear Regression | 2.4466 | 3.2795 | 0.0227 | 0.5152 | 0.5000 | 0.5833 | 0.5385 |
| 9988.HK | XGBoost | 2.5441 | 3.4042 | 0.0234 | 0.4747 | 0.4615 | 0.5000 | 0.4800 |
| 9988.HK | Random Forest | 2.6171 | 3.4697 | 0.0241 | 0.5556 | 0.5323 | 0.6875 | 0.6000 |

---

## 4. Limitations and Methodological Transparency

1. **Short-Term Horizon Limitations**: Daily stock price changes are largely dominated by intraday order flow and idiosyncratic news shocks, making next-day absolute price forecasting intrinsically noisy.
2. **Sentiment Data Window**: Collected social media and news discussions currently cover the latest 10-day monitoring window (~109 raw posts, 69 relevant posts). Because historical news and social discussions spanning the full 2-year backtest were not available without historical news APIs, models were primarily trained on technical OHLCV features and evaluated against the recent sentiment snapshot.
3. **No Financial Guarantee**: All ML and technical signals are experimental and intended for decision-support and educational demonstration only.
