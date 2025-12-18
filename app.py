# app.py
import streamlit as st
import bcrypt
import pandas as pd
from datetime import datetime
from scanner import run_scanner  # Import the function

# MUST BE FIRST
st.set_page_config(page_title="HA Daily Scanner", layout="wide")

# ==========================
# SIMPLE LOGIN (bcrypt + secrets)
# ==========================
def check_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🔐 Login Required")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == st.secrets["auth"]["user_name"]:
            stored_hash = st.secrets["auth"]["password"].encode()
            if bcrypt.checkpw(password.encode(), stored_hash):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid password")
        else:
            st.error("Invalid username")
    return False

if not check_login():
    st.stop()

# ==========================
# MAIN APP
# ==========================
st.title("📊 Heikin Ashi Daily Futures Scanner")
st.caption("End-of-Day | Risk-Aware | 18 Symbols")

if st.button("🔄 Run Scanner Now"):
    with st.spinner("Fetching data and calculating signals... (~20-30 seconds)"):
        report = run_scanner()

    if report.empty:
        st.info("No signals generated today.")
    else:
        st.success(f"Scan complete – {len(report)} results")

        # Filters
        col1, col2 = st.columns(2)
        verdict_options = report["Verdict"].unique()
        selected_verdicts = col1.multiselect("Verdict", verdict_options, default=["STRONG BUY", "STRONG SELL"])
        min_conf = col2.slider("Min Confidence %", 0, 100, 40)

        filtered = report[
            report["Verdict"].isin(selected_verdicts) &
            (report["Confidence_%"] >= min_conf)
        ]

        st.dataframe(report, use_container_width=True, hide_index=True)

        # Summary metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Signals", len(report))
        c2.metric("Strong Buys", (report["Verdict"] == "STRONG BUY").sum())
        c3.metric("Strong Sells", (report["Verdict"] == "STRONG SELL").sum())

        # Download
        csv = report.to_csv(index=False).encode()
        st.download_button(
            "📥 Download Report CSV",
            csv,
            f"HA_Scanner_{datetime.now().date()}.csv",
            "text/csv"
        )