# broker_api.py
# Dhan REST wrapper + instrument master helpers + simple DRY-RUN sim
# Strategy: used by main.py (Streamlit dashboard)

from __future__ import annotations
import os
import io
import time
import csv
import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from datetime import datetime, date
import requests
import pandas as pd
from dateutil import tz

# -----------------------------
# Dhan endpoints
# -----------------------------
DHAN_BASE_V2 = "https://api.dhan.co/v2"
# The market feed quote endpoint (v1) is commonly used to fetch LTP & volume
MARKET_FEED_QUOTE = "https://api.dhan.co/market-feed/v1.0/quote"  # POST
INSTRUMENT_CSV_URL = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"

# -----------------------------
# TZ helpers (IST)
# -----------------------------
IST = tz.gettz("Asia/Kolkata")

def now_ist() -> datetime:
    return datetime.now(tz=IST)

def minute_floor(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)

# -----------------------------
# Instrument Master Cache
# -----------------------------
class InstrumentCache:
    """Load and cache Dhan instrument master once per process."""
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.last_loaded: Optional[datetime] = None

    def load(self, force: bool = False) -> pd.DataFrame:
        if self.df is not None and not force:
            return self.df
        resp = requests.get(INSTRUMENT_CSV_URL, timeout=30)
        resp.raise_for_status()
        self.df = pd.read_csv(io.StringIO(resp.text))
        self.last_loaded = now_ist()
        return self.df

    def _ensure(self):
        if self.df is None:
            self.load()

    def nearest_expiry(self, underlying: str) -> str:
        """
        Return nearest expiry date (YYYY-MM-DD) for an index (NIFTY 50 / NIFTY BANK)
        scanning OPTIDX rows.
        """
        self._ensure()
        df = self.df
        u = underlying  # "NIFTY 50" or "NIFTY BANK"
        # Column names used in master (robust checks)
        col_under = "UNDERLYING_SYMBOL"
        col_type = "INSTRUMENT_TYPE"
        col_exp = "SM_EXPIRY_DATE"
        mask = (df.get(col_under) == u) & (df.get(col_type) == "OPTIDX")
        dd = df.loc[mask, col_exp].dropna().unique().tolist()
        # Keep only >= today
        today = date.today().isoformat()
        future = sorted([d for d in dd if str(d) >= today])
        return future[0] if future else sorted(dd)[-1]

    def index_security_id(self, underlying: str) -> Optional[int]:
        """
        Try to find the index 'INDEX' security id for NIFTY 50 / NIFTY BANK.
        """
        self._ensure()
        df = self.df
        col_sym = "SM_SYMBOL_NAME"
        col_type = "INSTRUMENT_TYPE"
        col_sid = "SECURITY_ID"
        try:
            m = (df.get(col_sym) == underlying) & (df.get(col_type).isin(["INDEX", "IDX"]))
            sid = df.loc[m, col_sid].dropna().astype(int)
            return int(sid.iloc[0]) if len(sid) else None
        except Exception:
            return None

    def option_security_ids(self, underlying: str, expiry: str, strike: int) -> Tuple[int, int, int]:
        """
        Return (CE_id, PE_id, lot_size) for given underlying (NIFTY 50 / NIFTY BANK), expiry (YYYY-MM-DD), strike.
        """
        self._ensure()
        df = self.df

        col_under = "UNDERLYING_SYMBOL"
        col_type = "INSTRUMENT_TYPE"
        col_exp = "SM_EXPIRY_DATE"
        col_strike = "STRIKE_PRICE"
        col_opt = "OPTION_TYPE"
        col_sid = "SECURITY_ID"
        # lot size column can be LOT_SIZE or LOT_UNITS
        col_lot = "LOT_SIZE" if "LOT_SIZE" in df.columns else ("SEM_LOT_UNITS" if "SEM_LOT_UNITS" in df.columns else None)

        strike_str = str(float(strike))
        base = (df.get(col_under) == underlying) & (df.get(col_type) == "OPTIDX") & (df.get(col_exp) == expiry) & (df.get(col_strike).astype(str) == strike_str)

        ce_id = df.loc[base & (df.get(col_opt) == "CE"), col_sid]
        pe_id = df.loc[base & (df.get(col_opt) == "PE"), col_sid]

        if ce_id.empty or pe_id.empty:
            raise RuntimeError(f"Cannot resolve CE/PE IDs for {underlying} {strike} {expiry}.")

        # lot size: take first any row in base
        lot_size = 1
        try:
            if col_lot and col_lot in df.columns:
                lot_val = df.loc[base, col_lot].dropna()
                if not lot_val.empty:
                    lot_size = int(float(lot_val.iloc[0]))
        except Exception:
            lot_size = 1

        return int(ce_id.iloc[0]), int(pe_id.iloc[0]), lot_size


_inst = InstrumentCache()

# -----------------------------
# DhanTrader (live) + SimTrader (dry run)
# -----------------------------
@dataclass
class Quote:
    ltp: Optional[float]
    total_qty: Optional[int]

class DhanTrader:
    """Minimal live wrapper used by the dashboard.
       All order calls are best-effort; focus is on live quotes."""
    def __init__(self, client_id: Optional[str] = None, access_token: Optional[str] = None):
        self.client_id = client_id or os.getenv("DHAN_CLIENT_ID", "")
        self.access_token = access_token or os.getenv("DHAN_ACCESS_TOKEN", "")
        if not self.client_id or not self.access_token:
            raise RuntimeError("Missing DHAN_CLIENT_ID or DHAN_ACCESS_TOKEN (set in Render Env Vars).")

    # ----- HTTP helpers
    def _headers(self) -> Dict[str, str]:
        return {
            "client-id": self.client_id,
            "access-token": self.access_token,
            "Content-Type": "application/json",
        }

    # ----- Quotes
    def quote(self, security_id: int, exchange_segment: str = "NSE_FNO") -> Quote:
        """Try a couple of segments to be robust."""
        segments = [exchange_segment, "NSE_INDEX", "NSE_CM", "NSE_FNO"]
        payload = {"securityId": str(security_id), "exchangeSegment": segments[0]}
        for seg in segments:
            payload["exchangeSegment"] = seg
            try:
                r = requests.post(MARKET_FEED_QUOTE, headers=self._headers(), json=payload, timeout=8)
                if r.status_code == 200:
                    j = r.json()
                    ltp = j.get("lastTradedPrice")
                    tq = j.get("totalTradedQuantity")
                    return Quote(ltp=float(ltp) if ltp is not None else None,
                                 total_qty=int(tq) if tq is not None else None)
            except Exception:
                pass
        return Quote(ltp=None, total_qty=None)

    # ----- Orders (best-effort; you can refine as needed)
    def place_market(self, transaction_type: str, security_id: int, quantity: int) -> Dict:
        """transaction_type: BUY / SELL; quantity int."""
        payload = {
            "dhanClientId": self.client_id,
            "transactionType": transaction_type,
            "exchangeSegment": "NSE_FNO",
            "productType": "INTRADAY",
            "orderType": "MARKET",
            "validity": "DAY",
            "securityId": str(security_id),
            "quantity": int(quantity),
        }
        try:
            r = requests.post(f"{DHAN_BASE_V2}/orders", headers=self._headers(), json=payload, timeout=10)
            if r.status_code >= 400:
                return {"error": r.text}
            return r.json()
        except Exception as e:
            return {"error": str(e)}

class SimTrader:
    """Pure local simulation (no network). We still use live IDs mapping from the CSV."""
    def __init__(self):
        self.positions: Dict[int, Dict] = {}  # security_id -> {qty, avg}
        self.prices: Dict[int, float] = {}

    def seed_price(self, security_id: int, ltp: float):
        self.prices[security_id] = ltp

    def quote(self, security_id: int, exchange_segment: str = "NSE_FNO") -> Quote:
        ltp = self.prices.get(security_id, None)
        return Quote(ltp=ltp, total_qty=None)

    def place_market(self, transaction_type: str, security_id: int, quantity: int) -> Dict:
        ltp = self.prices.get(security_id, 0.0)
        pos = self.positions.get(security_id, {"qty": 0, "avg": 0.0})
        if transaction_type == "BUY":
            # reduce shorts first
            if pos["qty"] < 0:
                # closing shorts
                new_qty = pos["qty"] + quantity
                pos["qty"] = new_qty
            else:
                # open long
                total_val = pos["avg"] * pos["qty"] + ltp * quantity
                pos["qty"] += quantity
                pos["avg"] = total_val / max(pos["qty"], 1)
        else:  # SELL
            if pos["qty"] > 0:
                pos["qty"] -= quantity
            else:
                # open short
                total_val = pos["avg"] * abs(pos["qty"]) + ltp * quantity
                pos["qty"] -= quantity
                pos["avg"] = total_val / max(abs(pos["qty"]), 1)
        self.positions[security_id] = pos
        return {"sim": True, "price": ltp, "qty": quantity, "side": transaction_type}

# -----------------------------
# Public helpers used by main.py
# -----------------------------
UNDERLYINGS = {
    "NIFTY": {"name": "NIFTY 50", "step": 50},
    "BANKNIFTY": {"name": "NIFTY BANK", "step": 100},
}

def round_to_step(x: float, step: int) -> int:
    return int(round(x / step) * step)

def resolve_locked_atm(index_key: str) -> Dict:
    """
    Resolve: nearest expiry + index security id + ATM strike + CE/PE ids + lot size.
    ATM is computed from index spot LTP at lock time (caller must call at/after 09:30 IST).
    """
    u = UNDERLYINGS[index_key]
    underlying = u["name"]

    # nearest expiry
    expiry = _inst.nearest_expiry(underlying)

    # try index spot id; if not found we fallback to using the closest strike from options after we fetch spot
    index_sid = _inst.index_security_id(underlying)

    # Build a temporary DhanTrader to get spot LTP if env vars are present; otherwise just return IDs without LTP
    spot_ltp = None
    try:
        t = DhanTrader()
        if index_sid:
            q = t.quote(index_sid, exchange_segment="NSE_INDEX")
            spot_ltp = q.ltp
    except Exception:
        pass

    # if we couldn't fetch spot, approximate later; but compute ATM now if ltp is available
    if spot_ltp is not None:
        atm = round_to_step(float(spot_ltp), u["step"])
    else:
        # fallback to the most common ATM (this will be corrected later when first quotes arrive)
        atm = round_to_step(underlying == "NIFTY 50" and 23000 or 50500, u["step"])

    ce_id, pe_id, lot_size = _inst.option_security_ids(underlying, expiry, atm)

    return {
        "index_key": index_key,
        "underlying": underlying,
        "expiry": expiry,
        "index_sid": index_sid,
        "atm": atm,
        "ce_id": ce_id,
        "pe_id": pe_id,
        "lot_size": lot_size,
        "spot_ltp": spot_ltp,
    }
