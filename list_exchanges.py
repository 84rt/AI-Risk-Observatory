import os
import requests
import json
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[0]
load_dotenv(REPO_ROOT / ".env.local")

API_BASE = "https://api.financialreports.eu"
API_KEY = os.environ.get("FR_API_KEY", "")
HEADERS = {"x-api-key": API_KEY, "Accept": "application/json"}

def list_exchanges():
    r = requests.get(f"{API_BASE}/exchanges/", headers=HEADERS)
    if r.status_code == 200:
        print(json.dumps(r.json(), indent=2))
    else:
        print(f"Error: {r.status_code} - {r.text}")

list_exchanges()
