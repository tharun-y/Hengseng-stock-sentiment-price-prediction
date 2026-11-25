
import yfinance as yf

ASSETS = {
    "HSBC": "0005.HK",
    "Tencent": "0700.HK",
    "Alibaba": "9988.HK",
    "Meituan": "3690.HK",
    "Xiaomi": "1810.HK",
}

print("Testing Hong Kong market data...\n")

for company, ticker in ASSETS.items():
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="10d", interval="1d")

        if data.empty:
            print(f"{company} ({ticker}): No data available")
            continue

        latest = data.iloc[-1]

        print(f"Company: {company}")
        print(f"Ticker: {ticker}")
        print(f"Latest Close: {latest['Close']:.2f}")
        print(f"Latest Date: {data.index[-1].date()}")
        print(f"Available Records: {len(data)}")
        print("-" * 40)

    except Exception as error:
        print(f"{company} ({ticker}): Error - {error}")
