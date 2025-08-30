import streamlit as st
import pandas as pd
import os
from broker_api import DhanTrader

# Load credentials from environment variables
ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
CLIENT_ID = os.getenv("DHAN_CLIENT_ID")

if not ACCESS_TOKEN or not CLIENT_ID:
    st.error("🚨 Missing Dhan credentials. Please set DHAN_ACCESS_TOKEN and DHAN_CLIENT_ID in Render Environment Variables.")
    st.stop()

# Initialize trader
trader = DhanTrader(access_token=ACCESS_TOKEN, client_id=CLIENT_ID)

# Streamlit App
st.set_page_config(page_title="Options Trading Bot", layout="wide")

st.title("📊 Options Trading Bot (Nifty & BankNifty)")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Mode", ["Paper Trading (Safe)", "Live Trading"])
lot_size = st.sidebar.number_input("Lots", min_value=1, value=1, step=1)
max_loss = st.sidebar.number_input("Max Loss (per lot)", min_value=500, value=2000, step=500)
start_strategy = st.sidebar.button("🚀 Start Strategy")
manual_exit = st.sidebar.button("❌ Exit All Positions")

# Dashboard placeholders
status_placeholder = st.empty()
pnl_placeholder = st.empty()
orders_placeholder = st.empty()

if start_strategy:
    status_placeholder.info("🔒 Locking ATM strike at 9:30... waiting for signal.")
    trader.start_strategy(lots=lot_size, max_loss=max_loss, live=(mode=="Live Trading"))

if manual_exit:
    trader.exit_all()

# Auto-update P&L table
if trader.active_positions:
    df = pd.DataFrame(trader.active_positions)
    pnl_placeholder.subheader("📈 Live Positions")
    pnl_placeholder.dataframe(df)
