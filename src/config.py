import os 
from dotenv import load_dotenv

load_dotenv()

APIFY_TOKEN = os.environ.get("APIFY_TOKEN")
EXA_API_KEY = os.environ.get("EXA_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def check_configuration():
    credentials = {
        "Apify": APIFY_TOKEN,
        "Exa": EXA_API_KEY,
        "OpenAI": OPENAI_API_KEY,
    }

    for service, key in credentials.items():
        if key and not key.endswith("_here"):
            print(f"{service}: Configured")
        else:
            print(f"{service}: Missing")

if __name__ == "__main__":
    check_configuration()