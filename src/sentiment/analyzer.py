
import os
import re
import json
import ast
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = "openai/gpt-oss-20b"

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. Add it to your project's .env file."
    )

client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# 2. FILE LOCATIONS
# ============================================================

SOCIAL_FILES = {
    "Telegram": (
        BASE_DIR / "src" / "collectors" / "data" /
        "raw" / "social" / "telegram_raw.csv"
    ),
    "LIHKG": (
        BASE_DIR / "src" / "collectors" / "data" /
        "raw" / "social" / "lihkg_raw.csv"
    )
}

NEWS_FILE = (
    BASE_DIR / "src" / "collectors" / "data" /
    "raw" / "news" / "news_articles.csv"
)

MARKET_FILE = (
    BASE_DIR / "src" / "collectors" / "data" /
    "raw" / "market" / "hong_kong_market_data.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. HONG KONG ASSETS AND COMPANY ALIASES
# ============================================================

ASSETS = {
    "0005.HK": {
        "company": "HSBC",
        "aliases": [
            "HSBC",
            "Hong Kong and Shanghai Banking Corporation",
            "滙豐",
            "汇丰"
        ]
    },

    "0700.HK": {
        "company": "Tencent",
        "aliases": [
            "Tencent",
            "騰訊",
            "腾讯"
        ]
    },

    "9988.HK": {
        "company": "Alibaba",
        "aliases": [
            "Alibaba",
            "阿里巴巴"
        ]
    },

    "3690.HK": {
        "company": "Meituan",
        "aliases": [
            "Meituan",
            "美團",
            "美团"
        ]
    },

    "1810.HK": {
        "company": "Xiaomi",
        "aliases": [
            "Xiaomi",
            "小米"
        ]
    }
}


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def clean_value(value):
    """Convert missing or invalid CSV values into clean strings."""

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def first_nonempty(row, columns):
    """Return the first available non-empty column value."""

    for column in columns:
        value = clean_value(row.get(column, ""))

        if value:
            return value

    return ""


def parse_number(value):
    """Safely convert engagement values into numbers."""

    try:
        value = clean_value(value).replace(",", "")

        if not value:
            return 0

        return float(value)

    except (ValueError, TypeError):
        return 0


def parse_list_value(value):
    """Handle lists stored as JSON or Python-style strings in CSVs."""

    if isinstance(value, list):
        return " ".join(str(item) for item in value)

    value = clean_value(value)

    if not value:
        return ""

    try:
        parsed = json.loads(value)

        if isinstance(parsed, list):
            return " ".join(str(item) for item in parsed)

    except (ValueError, TypeError):
        pass

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return " ".join(str(item) for item in parsed)

    except (ValueError, SyntaxError, TypeError):
        pass

    return value


def normalize_timestamp(value):
    """Convert timestamps into UTC ISO format."""

    value = clean_value(value)

    if not value:
        return ""

    timestamp = pd.to_datetime(
        value,
        utc=True,
        errors="coerce"
    )

    if pd.isna(timestamp):
        return ""

    return timestamp.isoformat()


def match_alias(text, alias):
    """Match English aliases without accidental partial matches."""

    if not text or not alias:
        return False

    if re.search(r"[A-Za-z0-9]", alias):
        pattern = r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9])"

        return bool(
            re.search(pattern, text, flags=re.IGNORECASE)
        )

    # Chinese names do not require word boundaries.
    return alias in text


def identify_tickers(row, source):
    """Identify all assets associated with a collected record."""

    if source == "News":
        direct_ticker = clean_value(row.get("ticker", ""))

        if direct_ticker in ASSETS:
            return [direct_ticker]

    search_fields = [
        "keyword",
        "matchedText",
        "source_query",
        "title",
        "text",
        "content",
        "own_text",
        "channelTitle"
    ]

    combined_text = " ".join(
        parse_list_value(row.get(field, ""))
        for field in search_fields
    )

    matched_tickers = []

    for ticker, asset in ASSETS.items():

        for alias in asset["aliases"]:

            if match_alias(combined_text, alias):
                matched_tickers.append(ticker)
                break

    return matched_tickers


# ============================================================
# 5. LOAD SOCIAL MEDIA DATA
# ============================================================

def load_social_data():

    all_records = []

    for source, file_path in SOCIAL_FILES.items():

        if not file_path.exists():
            print(f"WARNING: Missing file: {file_path}")
            continue

        df = pd.read_csv(file_path)

        print(f"{source}: Loaded {len(df)} records")

        for index, row in df.iterrows():

            row = row.to_dict()

            # Telegram messages
            if source == "Telegram":

                text = first_nonempty(
                    row,
                    ["text", "message", "messageText", "caption"]
                )

                timestamp = first_nonempty(
                    row,
                    ["date", "timestamp", "created_at"]
                )

                url = first_nonempty(
                    row,
                    ["url", "messageUrl"]
                )

                engagement = {
                    "views": parse_number(row.get("views")),
                    "reactions": parse_number(
                        row.get("reactionsTotal")
                    ),
                    "likes": 0,
                    "replies": 0
                }

                # This is the channel's subscriber count,
                # not an individual author's follower count.
                channel_followers = parse_number(
                    row.get("channelSubscribers")
                )

                author = first_nonempty(
                    row,
                    ["authorSignature", "author", "channelTitle"]
                )

            # LIHKG discussions
            else:

                text_parts = [
                    first_nonempty(row, ["title"]),
                    first_nonempty(row, ["content"]),
                    first_nonempty(row, ["own_text"])
                ]

                text = " ".join(
                    part for part in text_parts if part
                )

                timestamp = first_nonempty(
                    row,
                    ["created_at", "timestamp"]
                )

                url = first_nonempty(
                    row,
                    ["thread_url", "url"]
                )

                engagement = {
                    "views": 0,
                    "reactions": 0,
                    "likes": parse_number(row.get("like_count")),
                    "replies": parse_number(row.get("reply_count"))
                }

                channel_followers = None

                author = first_nonempty(
                    row,
                    ["author"]
                )

            if not text:
                continue

            tickers = identify_tickers(row, source)

            if not tickers:
                continue

            for ticker in tickers:

                all_records.append({
                    "id": f"{source}-{index}-{ticker}",
                    "source": source,
                    "ticker": ticker,
                    "company": ASSETS[ticker]["company"],
                    "timestamp": normalize_timestamp(timestamp),
                    "text": text[:4000],
                    "url": url,
                    "author": author,
                    "engagement": engagement,
                    "channel_followers": channel_followers
                })

    return all_records


# ============================================================
# 6. LOAD EXA NEWS DATA
# ============================================================

def load_news_data():

    if not NEWS_FILE.exists():
        print(f"WARNING: News file not found: {NEWS_FILE}")
        return []

    df = pd.read_csv(NEWS_FILE)

    print(f"Financial News: Loaded {len(df)} records")

    all_records = []

    for index, row in df.iterrows():

        row = row.to_dict()

        ticker = clean_value(row.get("ticker", ""))

        if ticker not in ASSETS:
            continue

        title = first_nonempty(row, ["title"])
        article_text = first_nonempty(row, ["text"])

        combined_text = " ".join(
            part for part in [title, article_text] if part
        )

        if not combined_text:
            continue

        all_records.append({
            "id": f"NEWS-{index}-{ticker}",
            "source": "News",
            "ticker": ticker,
            "company": ASSETS[ticker]["company"],
            "timestamp": normalize_timestamp(
                row.get("published_date")
            ),
            "text": combined_text[:4000],
            "url": first_nonempty(row, ["url"]),
            "author": first_nonempty(row, ["author"]),
            "engagement": {
                "views": 0,
                "reactions": 0,
                "likes": 0,
                "replies": 0
            },
            "channel_followers": None
        })

    return all_records


# ============================================================
# 7. GROQ SENTIMENT ANALYSIS
# ============================================================

SYSTEM_PROMPT = """
You are a financial sentiment analysis assistant.

Your task is to analyze social media posts and financial news
about Hong Kong listed companies.

The input may contain English or Chinese text.
Understand the meaning of both languages.

For each input record, determine:

1. sentiment:
   positive, negative, neutral, or irrelevant

2. sentiment_score:
   A number between -1 and +1.
   -1 means extremely negative.
    0 means neutral.
   +1 means extremely positive.

3. confidence:
   A number between 0 and 1.

4. direction:
   bullish, bearish, sideways, or unclear.

5. position:
   long, short, or none.
   Only report long or short if the text explicitly indicates
   that position. Never invent a trading position.

6. relevant:
   true if the content genuinely discusses the specified company
   or its stock in a financially meaningful way.
   false if it is an advertisement, unrelated mention, spam,
   or unrelated discussion.

7. reason:
   A short explanation of the classification.

Important rules:

- A company name appearing in an advertisement does not
  automatically make the content financially relevant.
- Do not assume positive sentiment just because a company
  is mentioned.
- Do not confuse general company news with stock direction.
- Do not invent financial facts or price levels.
- Do not invent support or resistance levels.
- If the content is irrelevant, use sentiment="irrelevant",
  sentiment_score=0, relevant=false, direction="unclear",
  and position="none".
- Return one result for every input ID.
- Preserve the exact input ID.

Return ONLY valid JSON in this structure:

{
  "results": [
    {
      "id": "input_id",
      "sentiment": "positive",
      "sentiment_score": 0.7,
      "confidence": 0.9,
      "direction": "bullish",
      "position": "none",
      "relevant": true,
      "reason": "Short explanation."
    }
  ]
}
"""


def parse_model_response(content):

    content = clean_value(content)

    # Remove Markdown code fences if the model adds them.
    content = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        content,
        flags=re.IGNORECASE
    )

    try:
        return json.loads(content)

    except json.JSONDecodeError:
        # Attempt to extract the outer JSON object.
        start = content.find("{")
        end = content.rfind("}")

        if start != -1 and end != -1:
            return json.loads(content[start:end + 1])

        raise ValueError("Groq returned invalid JSON.")


def analyze_batch(records):

    input_records = []

    for record in records:

        input_records.append({
            "id": record["id"],
            "ticker": record["ticker"],
            "company": record["company"],
            "source": record["source"],
            "text": record["text"]
        })

    user_prompt = f"""
Analyze the following financial posts and news articles.

Return a sentiment result for every input record.

INPUT RECORDS:

{json.dumps(input_records, ensure_ascii=False, indent=2)}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        response_format={
            "type": "json_object"
        },
        max_completion_tokens=1200
    )

    response_content = response.choices[0].message.content

    parsed = parse_model_response(response_content)

    return parsed.get("results", [])


# ============================================================
# 8. VALIDATE SENTIMENT RESULTS
# ============================================================

VALID_SENTIMENTS = {
    "positive",
    "negative",
    "neutral",
    "irrelevant"
}

VALID_DIRECTIONS = {
    "bullish",
    "bearish",
    "sideways",
    "unclear"
}

VALID_POSITIONS = {
    "long",
    "short",
    "none"
}


def normalize_result(result):

    sentiment = clean_value(
        result.get("sentiment", "neutral")
    ).lower()

    if sentiment not in VALID_SENTIMENTS:
        sentiment = "neutral"

    direction = clean_value(
        result.get("direction", "unclear")
    ).lower()

    if direction not in VALID_DIRECTIONS:
        direction = "unclear"

    position = clean_value(
        result.get("position", "none")
    ).lower()

    if position not in VALID_POSITIONS:
        position = "none"

    try:
        score = float(result.get("sentiment_score", 0))
    except (ValueError, TypeError):
        score = 0

    try:
        confidence = float(result.get("confidence", 0))
    except (ValueError, TypeError):
        confidence = 0

    score = max(-1, min(1, score))
    confidence = max(0, min(1, confidence))

    relevant = result.get("relevant", False)

    if isinstance(relevant, str):
        relevant = relevant.lower() == "true"
    else:
        relevant = bool(relevant)

    if not relevant or sentiment == "irrelevant":
        sentiment = "irrelevant"
        score = 0
        direction = "unclear"
        position = "none"
        relevant = False

    return {
        "sentiment": sentiment,
        "sentiment_score": score,
        "confidence": confidence,
        "direction": direction,
        "position": position,
        "relevant": relevant,
        "reason": clean_value(result.get("reason", ""))
    }


# ============================================================
# 9. SAVE RESULTS
# ============================================================

def save_results(results):

    if not results:
        print("No sentiment results to save.")
        return

    df = pd.DataFrame(results)

    csv_path = OUTPUT_DIR / "sentiment_results.csv"
    json_path = OUTPUT_DIR / "sentiment_results.json"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    json_records = (
        df.astype(object)
        .where(pd.notna(df), None)
        .to_dict(orient="records")
    )

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            json_records,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"Saved sentiment CSV: {csv_path}")
    print(f"Saved sentiment JSON: {json_path}")


# ============================================================
# 10. CREATE TICKER-WISE SUMMARY
# ============================================================

def create_summary(results):

    if not results:
        return

    df = pd.DataFrame(results)

    summary_rows = []

    for ticker, asset in ASSETS.items():

        ticker_df = df[df["ticker"] == ticker]

        relevant_df = ticker_df[
            ticker_df["relevant"] == True
        ]

        if len(relevant_df) > 0:

            average_score = relevant_df[
                "sentiment_score"
            ].mean()

        else:
            average_score = 0

        summary_rows.append({

            "ticker": ticker,
            "company": asset["company"],

            "total_mentions": len(ticker_df),

            "relevant_mentions": len(relevant_df),

            "positive_mentions": int(
                (relevant_df["sentiment"] == "positive").sum()
            ),

            "negative_mentions": int(
                (relevant_df["sentiment"] == "negative").sum()
            ),

            "neutral_mentions": int(
                (relevant_df["sentiment"] == "neutral").sum()
            ),

            "average_sentiment_score": round(
                float(average_score), 4
            ),

            "bullish_mentions": int(
                (relevant_df["direction"] == "bullish").sum()
            ),

            "bearish_mentions": int(
                (relevant_df["direction"] == "bearish").sum()
            ),

            "sideways_mentions": int(
                (relevant_df["direction"] == "sideways").sum()
            )
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df.sort_values(
        by="average_sentiment_score",
        ascending=False
    )

    summary_path = OUTPUT_DIR / "sentiment_summary.csv"

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nSaved ticker summary: {summary_path}")

    print("\n========== SENTIMENT SUMMARY ==========")

    print(summary_df.to_string(index=False))


# ============================================================
# 11. MAIN EXECUTION
# ============================================================

def main():

    print("\n========================================")
    print(" HONG KONG STOCK SENTIMENT ANALYZER")
    print(" Powered by Groq Cloud")
    print("========================================\n")

    if MARKET_FILE.exists():
        market_df = pd.read_csv(MARKET_FILE)
        print(f"Market data available: {len(market_df)} records")
    else:
        print("WARNING: Market data file not found.")

    social_records = load_social_data()
    news_records = load_news_data()

    all_records = social_records + news_records

    print("\n========== DATA COLLECTION SUMMARY ==========")
    print(f"Telegram + LIHKG records: {len(social_records)}")
    print(f"Financial news records: {len(news_records)}")
    print(f"Total records: {len(all_records)}")

    if not all_records:
        print("No records found. Check your CSV paths.")
        return

    # Resume from previously saved results.
    results_file = OUTPUT_DIR / "sentiment_results.csv"

    all_results = []

    if results_file.exists():
        existing_df = pd.read_csv(results_file)

        all_results = existing_df.to_dict(orient="records")

        print(
            f"\nPreviously completed records: {len(all_results)}"
        )

    processed_ids = {
        clean_value(result.get("id"))
        for result in all_results
    }

    pending_records = [
        record for record in all_records
        if record["id"] not in processed_ids
    ]

    print(f"Already processed: {len(processed_ids)}")
    print(f"Remaining records: {len(pending_records)}")

    if not pending_records:
        print("\nAll records have already been analyzed.")
        create_summary(all_results)
        return

    # Smaller batches reduce token consumption.
    BATCH_SIZE = 3

    total_batches = (
        len(pending_records) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    print("\nStarting Groq sentiment analysis...\n")

    for start in range(0, len(pending_records), BATCH_SIZE):

        batch = pending_records[start:start + BATCH_SIZE]

        batch_number = (start // BATCH_SIZE) + 1

        print(
            f"Processing batch {batch_number}/{total_batches} "
            f"({len(batch)} records)"
        )

        try:

            model_results = analyze_batch(batch)

            results_by_id = {
                clean_value(item.get("id")): item
                for item in model_results
            }

            for record in batch:

                model_result = results_by_id.get(
                    record["id"], {}
                )

                if not model_result:
                    print(
                        f"WARNING: Missing model result for "
                        f"{record['id']}"
                    )

                validated = normalize_result(model_result)

                final_record = {
                    **record,
                    **validated
                }

                all_results.append(final_record)

            print(f"Batch {batch_number} completed.")

            # Save progress after every successful batch.
            save_results(all_results)

        except Exception as error:

            save_results(all_results)

            print("\nGroq API error:", error)

            print(
                "\nYour completed results have been saved. "
                "Run this script again later to resume."
            )

            return

    create_summary(all_results)

    print("\n========================================")
    print(" SENTIMENT ANALYSIS COMPLETED")
    print("========================================")

    print(f"Total analyzed records: {len(all_results)}")

    relevant_count = sum(
        1 for result in all_results
        if str(result.get("relevant", "")).lower() == "true"
    )

    print(f"Relevant records: {relevant_count}")
    print(f"Irrelevant records: {len(all_results) - relevant_count}")

    print(f"\nOutput directory: {OUTPUT_DIR}")



if __name__ == "__main__":
    main()
