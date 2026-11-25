
import os
import json
import pandas as pd

from pathlib import Path
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from exa_py import Exa


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("EXA_API_KEY")

if not API_KEY:
    raise ValueError("EXA_API_KEY not found in .env file")

exa = Exa(api_key=API_KEY)

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "src" / "collectors" / "data" / "raw" / "news"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = {
    "0005.HK": "HSBC Holdings",
    "0700.HK": "Tencent Holdings",
    "9988.HK": "Alibaba Group",
    "3690.HK": "Meituan",
    "1810.HK": "Xiaomi Corporation",
}

DAYS = 10


# --------------------------------------------------
# NEWS COLLECTION
# --------------------------------------------------

def collect_news(ticker, company):

    print(f"\nSearching news for {company} ({ticker})...")

    query = (
        f"Latest financial news, stock market updates, "
        f"earnings, business developments and analyst opinions "
        f"about {company} Hong Kong stock {ticker}"
    )

    start_date = (
        datetime.now(timezone.utc) - timedelta(days=DAYS)
    ).date().isoformat()

    try:

        response = exa.search(
            query,
            type="auto",
            category="news",
            num_results=10,
            start_published_date=start_date,
            contents={
                "text": {
                    "max_characters": 5000
                }
            }
        )

        articles = []

        for result in response.results:

            article = {
                "ticker": ticker,
                "company": company,
                "title": result.title,
                "published_date": result.published_date,
                "author": result.author,
                "url": result.url,
                "text": result.text,
                "source": "Exa"
            }

            articles.append(article)

        print(f"Collected {len(articles)} articles for {company}")

        return articles

    except Exception as e:

        print(f"Error collecting news for {company}: {e}")

        return []


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    all_articles = []

    for ticker, company in ASSETS.items():

        articles = collect_news(ticker, company)

        all_articles.extend(articles)

    if not all_articles:

        print("\nNo news articles collected.")
        return

    df = pd.DataFrame(all_articles)

    # Remove duplicate articles
    df = df.drop_duplicates(subset=["url"])

    # Save CSV
    csv_path = OUTPUT_DIR / "news_articles.csv"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Save JSON
    json_path = OUTPUT_DIR / "news_articles.json"

    df.to_json(
        json_path,
        orient="records",
        indent=4,
        force_ascii=False
    )

    print("\n----------------------------------")
    print("NEWS COLLECTION COMPLETED")
    print("----------------------------------")

    print(f"Total articles: {len(df)}")
    print(f"CSV saved: {csv_path}")
    print(f"JSON saved: {json_path}")

    print("\nArticles collected per company:")

    print(
        df.groupby(["ticker", "company"])
        .size()
        .to_string()
    )


if __name__ == "__main__":
    main()
