# main.py
# Streamlit dashboard for VWAP Reversal Straddle (combined premium),
# with ATM lock at 09:30 IST, first-index trigger, per-leg exit/re-entry,
# manual square-off, DRY_RUN vs LIVE toggle.

from __future__ import annotations
import os
import time
from datetime import datetime, time as dtime
from typing import Dict, List, Optional

import streamlit as st
import pandas as pd
from dateutil import tz

from broker_api import (
    DhanTrader,
    SimTrader,
    UNDERLYINGS,
    resolve_locked_atm,
    round_to_step,
    now_ist,
    minute_floor,
)

IST = tz.gettz("Asia/Kolkata")

# -----------------------------
# Page & sidebar controls
# -----------------------------
st.set_page_config(page_title="VWAP Reversal Straddle", layout="wide")
st.title("📈 VWAP Reversal Straddle — NIFTY / BANKNIFTY")

with st.sidebar:
    st.subheader("⚙️ Controls")
    live_trading = st.toggle("LIVE Trading", value=False, help="Off = DRY RUN (safe). On = send orders to Dhan.")
    lots = st.number_input("Lots", min_value=1, value=1, step=1)
    per_lot_sl = st.number_input("Per-leg SL (₹ per lot)", min_value=100, value=500, step=100)
    max_day_loss = st.number_input("Max day loss (₹ per lot)", min_value=0, value=0, step=500, help="0 to disable")
    refresh_sec = st.slider("Refresh interval (sec)", 2, 10, 5)

    st.divider()
    st.caption("🔐 Credentials (for LIVE). Keep set as Render environment variables.")
    st.text_input("DHAN_CLIENT_ID", value=os.getenv("DHAN_CLIENT_ID", ""), disabled=True)
    st.text_input("DHAN_ACCESS_TOKEN", value=os.getenv("DHAN_ACCESS_TOKEN", ""), type="password", disabled=True)

    st.divider()
    start_btn = st.button("▶️ Start Strategy", type="primary")
    stop_btn = st.button("⏹️ Stop Strategy")
    squareoff_btn = st.button("🧹 Square-off Now")

# -----------------------------
# Session state init
# -----------------------------
def _init_state():
    st.session_state.state = {
        "running": False,
        "locked": {},  # per index: resolved dict with ids & atm
        "chosen": None,  # index_key chosen for today (first to trigger)
        "legs": [],  # [{"type": "CE"/"PE", "sid": int, "qty": int, "entry": float, "open": True}]
        "series": {  # per index series
            "NIFTY": {"ts": [], "cp": [], "vol": [], "vwap": [], "last_tq": {"ce": None, "pe": None}},
            "BANKNIFTY": {"ts": [], "cp": [], "vol": [], "vwap": [], "last_tq": {"ce": None, "pe": None}},
        },
        "realized": 0.0,
        "entry_time": None,
        "locked_at": None,
    }

if "state" not in st.session_state:
    _init_state()

S = st.session_state.state

# choose broker
broker = DhanTrader() if live_trading else SimTrader()

# -----------------------------
# Strategy constants
# -----------------------------
ENTRY_TIME = dtime(9, 30)
EXIT_TIME = dtime(14, 55)

def qty_for_lots(index_key: str, lots: int) -> int:
    # We already resolved lot_size when locking ATM
    lot = S["locked"].get(index_key, {}).get("lot_size", 1)
    return lot * lots

# -----------------------------
# Helpers
# -----------------------------
def per_minute_sample(index_key: str) -> Optional[Dict]:
    """
    Grab CE+PE quotes, compute combined premium and per-minute delta volume,
    update VWAP series. Requires we already locked CE/PE IDs.
    """
    if index_key not in S["locked"]:
        return None
    ce_id = S["locked"][index_key]["ce_id"]
    pe_id = S["locked"][index_key]["pe_id"]

    # get quotes
    q_ce = broker.quote(ce_id, "NSE_FNO")
    q_pe = broker.quote(pe_id, "NSE_FNO")
    if q_ce.ltp is None or q_pe.ltp is None:
        return None

    # Track total traded qty to get per-minute delta
    last_tq = S["series"][index_key]["last_tq"]
    ce_vol = 0
    pe_vol = 0
    if q_ce.total_qty is not None:
        ce_vol = 0 if last_tq["ce"] is None else max(q_ce.total_qty - last_tq["ce"], 0)
        last_tq["ce"] = q_ce.total_qty
    if q_pe.total_qty is not None:
        pe_vol = 0 if last_tq["pe"] is None else max(q_pe.total_qty - last_tq["pe"], 0)
        last_tq["pe"] = q_pe.total_qty

    combined = float(q_ce.ltp) + float(q_pe.ltp)
    volume = (ce_vol or 0) + (pe_vol or 0)
    ts = minute_floor(now_ist())

    # append
    ser = S["series"][index_key]
    ser["ts"].append(ts)
    ser["cp"].append(combined)
    ser["vol"].append(volume if volume is not None else 0)

    # compute running VWAP on combined premium
    vals = []
    vols = []
    csum = 0.0
    vtotal = 0
    for p, v in zip(ser["cp"], ser["vol"]):
        v = int(v) if v is not None else 0
        csum += p * max(v, 1)  # fallback weight 1 if no volume tick
        vtotal += max(v, 1)
        vals.append(csum / vtotal if vtotal else p)
        vols.append(v)
    ser["vwap"] = vals

    return {"ts": ts, "combined": combined, "v": volume, "vwap": vals[-1] if vals else combined}

def pattern_entry(index_key: str) -> bool:
    """
    Entry when last two closes ABOVE vwap then two closes BELOW vwap (2↑ then 2↓).
    """
    ser = S["series"][index_key]
    if len(ser["cp"]) < 4:
        return False
    p = ser["cp"][-4:]
    v = ser["vwap"][-4:]
    return (p[0] > v[0] and p[1] > v[1] and p[2] < v[2] and p[3] < v[3])

def pattern_exit_above(index_key: str) -> bool:
    """
    Exit condition VWAP part: last two closes ABOVE vwap (2↑).
    """
    ser = S["series"][index_key]
    if len(ser["cp"]) < 2:
        return False
    p = ser["cp"][-2:]
    v = ser["vwap"][-2:]
    return (p[0] > v[0] and p[1] > v[1])

# -----------------------------
# ATM lock at/after 09:30 IST
# -----------------------------
def ensure_locked():
    if S["locked"]:
        return
    nowt = now_ist().time()
    if nowt < ENTRY_TIME:
        st.info("⏳ Waiting until 09:30 IST to lock ATM.")
        return
    # lock both indices
    for k in ["NIFTY", "BANKNIFTY"]:
        try:
            info = resolve_locked_atm(k)
            S["locked"][k] = info
        except Exception as e:
            st.error(f"Failed to lock ATM for {k}: {e}")
    S["locked_at"] = minute_floor(now_ist())

# -----------------------------
# Order helpers
# -----------------------------
def place_straddle(index_key: str):
    """SELL CE & SELL PE on locked ATM for chosen index."""
    info = S["locked"][index_key]
    lot_qty = qty_for_lots(index_key, lots)
    legs = []
    # place SELL CE/PE (market)
    for typ, sid in (("CE", info["ce_id"]), ("PE", info["pe_id"])):
        if isinstance(broker, SimTrader):
            # seed current price for sim to avoid None
            q = broker.quote(sid)
            if q.ltp is None:
                broker.seed_price(sid, 100.0)
        res = broker.place_market("SELL", sid, lot_qty)
        q = broker.quote(sid)
        entry = float(q.ltp or 0.0)
        legs.append({"type": typ, "sid": sid, "qty": lot_qty, "entry": entry, "open": True, "reenter_ok": True})
    S["legs"] = legs
    S["entry_time"] = minute_floor(now_ist())

def square_off_all():
    for leg in S["legs"]:
        if leg["open"]:
            broker.place_market("BUY", leg["sid"], leg["qty"])
            # realized pnl update
            q = broker.quote(leg["sid"])
            exitp = float(q.ltp or 0.0)
            # short option P&L: (entry - exit) * qty
            S["realized"] += (leg["entry"] - exitp) * leg["qty"]
            leg["open"] = False

# -----------------------------
# Start/Stop/UI triggers
# -----------------------------
if stop_btn and S["running"]:
    square_off_all()
    S["running"] = False
    st.warning("🔴 Stopped and squared off.")

if squareoff_btn:
    square_off_all()
    st.info("🧹 Manual square-off sent.")

if start_btn and not S["running"]:
    ensure_locked()
    if not S["locked"]:
        st.warning("Cannot start until ATM is locked.")
    else:
        S["running"] = True
        S["chosen"] = None
        st.success("✅ Strategy started. Waiting for first index to trigger…")

# -----------------------------
# Main loop when running
# -----------------------------
def leg_unrealized(leg) -> float:
    q = broker.quote(leg["sid"])
    ltp = float(q.ltp or 0.0)
    # short P&L
    return (leg["entry"] - ltp) * leg["qty"]

def total_open_unrealized() -> float:
    return sum(leg_unrealized(l) for l in S["legs"] if l["open"])

def per_leg_loss_exceeds(leg) -> bool:
    loss = -min(0.0, leg_unrealized(leg))  # positive loss
    threshold = per_lot_sl * lots
    return loss > threshold

# Header KPIs
k1, k2, k3, k4 = st.columns(4)
if S["locked"]:
    for k in ["NIFTY", "BANKNIFTY"]:
        info = S["locked"].get(k)
        if info:
            q_ce = broker.quote(info["ce_id"])
            q_pe = broker.quote(info["pe_id"])
            locked_at = S.get("locked_at")
            st.caption(f"🔒 {k} ATM {info['atm']} (locked {locked_at.strftime('%H:%M') if locked_at else ''}) | CE LTP {q_ce.ltp} | PE LTP {q_pe.ltp}")

if S["running"]:
    ensure_locked()
    # sample both indices
    for idx in ["NIFTY", "BANKNIFTY"]:
        if idx in S["locked"]:
            per_minute_sample(idx)

    # choose first that triggers
    if S["chosen"] is None:
        for idx in ["NIFTY", "BANKNIFTY"]:
            if pattern_entry(idx):
                S["chosen"] = idx
                place_straddle(idx)
                st.success(f"🚀 Entry: {idx} ATM {S['locked'][idx]['atm']} @ {S['entry_time'].strftime('%H:%M')}")
                break

    # manage exits / re-entries
    chosen = S["chosen"]
    if chosen:
        # per-leg exit rule (loss > ₹500/lot & last two closes above VWAP)
        if pattern_exit_above(chosen):
            for leg in S["legs"]:
                if leg["open"] and per_leg_loss_exceeds(leg):
                    broker.place_market("BUY", leg["sid"], leg["qty"])
                    q = broker.quote(leg["sid"])
                    exitp = float(q.ltp or 0.0)
                    S["realized"] += (leg["entry"] - exitp) * leg["qty"]
                    leg["open"] = False
                    st.warning(f"⚠️ Exit leg {leg['type']} due to SL + VWAP condition.")

        # re-entry rule: if exited leg is allowed and last two closes BELOW vwap
        ser = S["series"][chosen]
        if len(ser["cp"]) >= 2:
            below2 = ser["cp"][-2] < ser["vwap"][-2] and ser["cp"][-1] < ser["vwap"][-1]
        else:
            below2 = False

        if below2:
            for leg in S["legs"]:
                if not leg["open"] and leg.get("reenter_ok", True):
                    # re-enter SELL
                    res = broker.place_market("SELL", leg["sid"], leg["qty"])
                    q = broker.quote(leg["sid"])
                    leg["entry"] = float(q.ltp or 0.0)
                    leg["open"] = True
                    leg["reenter_ok"] = False  # avoid spam multiple times
                    st.info(f"🔁 Re-entered leg {leg['type']} after VWAP condition.")

    # final exit at 14:55
    if now_ist().time() >= EXIT_TIME and any(l["open"] for l in S["legs"]):
        square_off_all()
        S["running"] = False
        st.warning("⏱️ Final exit at 14:55 — squared off all.")

    # day risk cap
    if max_day_loss and (-(S["realized"] + total_open_unrealized()) > max_day_loss * lots):
        square_off_all()
        S["running"] = False
        st.error("🛑 Max day loss hit — halted & squared off.")

    # UI table
    st.subheader("Positions")
    rows = []
    for leg in S["legs"]:
        u = leg_unrealized(leg)
        rows.append({
            "Leg": leg["type"],
            "Open": leg["open"],
            "Qty": leg["qty"],
            "Entry": round(leg["entry"], 2),
            "Unrealized (₹)": round(u, 2),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.metric("Unrealized P&L (₹)", value=f"{round(total_open_unrealized(),2)}")
    st.metric("Realized P&L (₹)", value=f"{round(S['realized'],2)}")

    time.sleep(refresh_sec)
    st.rerun()
else:
    st.info("Strategy is idle. Set lots/SL, choose LIVE or keep DRY RUN, then click **Start Strategy**.")
