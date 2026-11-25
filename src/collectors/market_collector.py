
import yfinance as yf
import pandas as pd
from pathlib import Path

# Five Hong Kong assets
ASSETS = {
    "HSBC": "0005.HK",
    "Tencent": "0700.HK",
    "Alibaba": "9988.HK",
    "Meituan": "3690.HK",
    "Xiaomi": "1810.HK",
}

# Output directory
OUTPUT_DIR = Path("data/raw/market")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def collect_market_data():

    all_data = []

    for company, ticker in ASSETS.items():

        print(f"\nCollecting {company} ({ticker})...")

        try:
            stock = yf.Ticker(ticker)

            # Two years of daily historical data
            data = stock.history(
                period="2y",
                interval="1d",
                auto_adjust=True
            )

            if data.empty:
                print(f"No data available for {ticker}")
                continue

            # Convert index to UTC
            if data.index.tz is not None:
                data.index = data.index.tz_convert("UTC")

            data = data.reset_index()

            # Standardize columns
            data = data.rename(columns={
                "Date": "timestamp",
                "Datetime": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            })

            data["company"] = company
            data["ticker"] = ticker

            # Keep required columns
            data = data[
                [
                    "timestamp",
                    "company",
                    "ticker",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume"
                ]
            ]

            all_data.append(data)

            # Save individual asset data
            output_file = OUTPUT_DIR / f"{ticker.replace('.', '_')}.csv"
            data.to_csv(output_file, index=False)

            print(f"Records collected: {len(data)}")
            print(f"Latest closing price: {data.iloc[-1]['close']:.2f}")
            print(f"Saved: {output_file}")

        except Exception as error:
            print(f"Error collecting {ticker}: {error}")

    if all_data:

        combined_data = pd.concat(all_data, ignore_index=True)

        combined_file = OUTPUT_DIR / "hong_kong_market_data.csv"
        combined_data.to_csv(combined_file, index=False)

        print("\n" + "=" * 50)
        print("MARKET DATA COLLECTION COMPLETED")
        print("=" * 50)
        print("Total records:", len(combined_data))
        print("Assets collected:", combined_data["ticker"].nunique())
        print("Combined file:", combined_file)

    else:
        print("No market data was collected.")


if __name__ == "__main__":
    collect_market_data()
