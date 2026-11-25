
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

FILES = {
    "Social Media_Telegram": BASE_DIR / "src" / "collectors" / "data" / "raw" / "social" / "telegram_raw.csv",
    "Social Media_lihkg": BASE_DIR / "src" / "collectors" / "data" / "raw" / "social" / "lihkg_raw.csv",
    "Financial News": BASE_DIR / "src" / "collectors" / "data" / "raw" / "news" / "news_articles.csv",
    "Market Data": BASE_DIR / "src" / "collectors" / "data" / "raw" / "market" / "hong_kong_market_data.csv"
}

for name, path in FILES.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    if not path.exists():
        print(f"File not found: {path}")
        continue

    df = pd.read_csv(path)

    print("\nTotal records:", len(df))

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 3 records:")
    print(df.head(3).to_string(index=False))

    print("\nMissing values:")
    print(df.isnull().sum().to_string())
