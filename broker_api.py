# broker_api.py
import requests
from config import DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN

BASE_URL = "https://api.dhan.co/v2"

def _headers():
    return {
        "Content-Type": "application/json",
        "client-id": DHAN_CLIENT_ID,
        "access-token": DHAN_ACCESS_TOKEN,
    }

def place_order(payload):
    url = f"{BASE_URL}/orders"
    resp = requests.post(url, headers=_headers(), json=payload)
    return resp.json()

def square_off(order_id):
    url = f"{BASE_URL}/orders/{order_id}/cancel"
    resp = requests.delete(url, headers=_headers())
    return resp.json()
