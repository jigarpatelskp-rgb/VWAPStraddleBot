# config.py
import os
from dotenv import load_dotenv

# Load .env file in local dev (not needed in Render since it uses Env Vars)
load_dotenv()

# --- DHAN Credentials (loaded safely) ---
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "")

# --- Default Settings ---
DEFAULT_MODE = "paper"   # "paper" or "live"
DEFAULT_LOTS = 1
MAX_LOSS = 2000          # per day stoploss in ₹
