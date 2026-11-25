
import os
import json
import pandas as pd

from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
from apify_client import ApifyClient

load_dotenv()

APIFY_TOKEN = os.getenv("APIFY_TOKEN")

if not APIFY_TOKEN:
    raise ValueError("APIFY_TOKEN is missing")

client = ApifyClient(APIFY_TOKEN)

# Five Hong Kong stocks
ASSETS = {
    "HSBC": {
        "ticker": "0005.HK",
        "keywords": ["HSBC", "0005.HK", "滙豐"]
    },
    "Tencent": {
        "ticker": "0700.HK",
        "keywords": ["Tencent", "0700.HK", "騰訊"]
    },
    "Alibaba": {
        "ticker": "9988.HK",
        "keywords": ["Alibaba", "9988.HK", "阿里巴巴"]
    },
    "Meituan": {
        "ticker": "3690.HK",
        "keywords": ["Meituan", "3690.HK", "美團"]
    },
    "Xiaomi": {
        "ticker": "1810.HK",
        "keywords": ["Xiaomi", "1810.HK", "小米"]
    }
}

LIHKG_ACTOR = "claude_code_reviewer/lihkg-scraper-en"
TELEGRAM_ACTOR = "k-data-lab/telegram-channel-scraper"

OUTPUT_DIR = Path("data/raw/social")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_record(record, source):

    if source == "LIHKG":
        text = (
            record.get("own_text")
            or record.get("content")
            or record.get("title")
            or ""
        )

        timestamp = (
            record.get("created_at")
            or record.get("last_reply_at")
        )

        likes = record.get("like_count")
        views = None
        reposts = None
        followers = None

        author = record.get("author")
        url = record.get("thread_url")
        keyword = record.get("source_query")

    else:
        text = record.get("text") or ""
        timestamp = record.get("date")

        likes = record.get("reactionsTotal")
        views = record.get("views")
        reposts = None
        followers = record.get("channelSubscribers")

        author = (
            record.get("author")
            or record.get("channelUsername")
        )

        url = record.get("url")
        keyword = record.get("keyword")

    return {
        "source": source,
        "keyword": keyword,
        "timestamp": timestamp,
        "text": text,
        "author": author,
        "author_followers": followers,
        "likes": likes,
        "reposts": reposts,
        "views": views,
        "url": url
    }


def collect_lihkg():

    print("\nStarting LIHKG collection...")

    keywords = [
        keyword
        for asset in ASSETS.values()
        for keyword in asset["keywords"]
    ]

    run = client.actor(LIHKG_ACTOR).call(
        run_input={
            "mode": "search",
            "keywords": keywords,
            "order": "desc_create_time",
            "dateFrom": "10d",
            "maxItems": 10,
            "aiLabels": False
        }
    )

    if not run:
        print("LIHKG actor did not return a run.")
        return []

    items = client.dataset(
        run.default_dataset_id
    ).list_items().items

    print("LIHKG records collected:", len(items))

    return [
        normalize_record(item, "LIHKG")
        for item in items
    ]


def collect_telegram():

    print("\nStarting Telegram collection...")

    keywords = [
        keyword
        for asset in ASSETS.values()
        for keyword in asset["keywords"]
    ]

    run = client.actor(TELEGRAM_ACTOR).call(
        run_input={
            "keywords": keywords,
            "since": "10 days",
            "maxResults": 50,
            "maxMessagesPerChannel": 10
        }
    )

    if not run:
        print("Telegram actor did not return a run.")
        return []

    items = client.dataset(
        run.default_dataset_id
    ).list_items().items

    print("Telegram records collected:", len(items))

    return [
        normalize_record(item, "Telegram")
        for item in items
    ]


def main():

    all_records = []

    try:
        all_records.extend(collect_lihkg())
    except Exception as error:
        print("LIHKG error:", error)

    try:
        all_records.extend(collect_telegram())
    except Exception as error:
        print("Telegram error:", error)

    if not all_records:
        print("\nNo social records collected.")
        return

    df = pd.DataFrame(all_records)

    # Convert timestamps into UTC
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    # Keep records from the last 10 days
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=10)

    df = df[
        (df["timestamp"] >= cutoff)
        & (df["timestamp"] <= pd.Timestamp.now(tz="UTC"))
    ].copy()

    # Match records to the five assets
    def identify_asset(row):

        search_text = (
            str(row["text"]) + " " + str(row["keyword"])
        ).lower()

        for company, details in ASSETS.items():
            for keyword in details["keywords"]:
                if keyword.lower() in search_text:
                    return pd.Series([
                        company,
                        details["ticker"]
                    ])

        return pd.Series([None, None])

    df[["asset", "ticker"]] = df.apply(
        identify_asset,
        axis=1
    )

    df = df.dropna(subset=["asset"])

    # Save combined data
    csv_file = OUTPUT_DIR / "social_posts.csv"
    json_file = OUTPUT_DIR / "social_posts.json"

    df.to_csv(csv_file, index=False)

    df.to_json(
        json_file,
        orient="records",
        indent=2,
        date_format="iso"
    )

    print("\n" + "=" * 45)
    print("SOCIAL DATA COLLECTION COMPLETED")
    print("=" * 45)

    print("Total posts:", len(df))
    print("\nPosts by source:")
    print(df["source"].value_counts())

    print("\nPosts by asset:")
    print(df["asset"].value_counts())

    print("\nSaved:", csv_file)
    print("Saved:", json_file)


if __name__ == "__main__":
    main()
