
from apify_client import ApifyClient
from dotenv import load_dotenv
import os

load_dotenv()

APIFY_TOKEN = os.getenv("APIFY_TOKEN")

if not APIFY_TOKEN:
    raise ValueError("APIFY_TOKEN is missing from your .env file")

client = ApifyClient(APIFY_TOKEN)

try:
    user = client.user().get()

    if user:
        print("Apify connection successful!")
        print("Account:", user.username)
        print("User ID:", user.id)
    else:
        print("Could not retrieve Apify account details.")

except Exception as error:
    print("Apify connection failed.")
    print("Error:", error)
