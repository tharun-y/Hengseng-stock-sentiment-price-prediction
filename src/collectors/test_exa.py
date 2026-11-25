import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from exa_py import Exa

load_dotenv()

api_key = os.getenv("EXA_API_KEY")

if not api_key:
    raise ValueError("EXA_API_KEY is missing from your .env file")

exa = Exa(api_key=api_key)

try:
    ten_days_ago = (
        datetime.now(timezone.utc) - timedelta(days=10)
    ).strftime("%Y-%m-%d")

    print("Searching recent financial news...")

    response = exa.search(
        "Latest financial news about Hang Seng Index and Hong Kong stocks",
        type="auto",
        num_results=3,
        start_published_date=ten_days_ago,
        contents={"highlights": True}
    )

    print("\nExa connection successful!")
    print(f"Articles retrieved: {len(response.results)}")

    for index, article in enumerate(response.results, start=1):
        print(f"\n--- Article {index} ---")
        print("Title:", article.title)
        print("URL:", article.url)
        print("Published:", article.published_date)
        print("Highlights:", article.highlights)

except Exception as error:
    print("\nExa request failed.")
    print("Error:", error)