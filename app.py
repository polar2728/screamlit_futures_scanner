import streamlit as st
import bcrypt
import pandas as pd
from datetime import datetime, date
from scanner import run_scanner

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="HA Daily Scanner", layout="wide", initial_sidebar_state="expanded")

# ==========================
# LOGIN
# ==========================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Login to HA Scanner")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if username == st.secrets["auth"]["user_name"]:
                stored_hash = st.secrets["auth"]["password"].encode()
                if bcrypt.checkpw(password.encode(), stored_hash):
                    st.session_state.authenticated = True
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid password")
            else:
                st.error("Invalid username")
    st.stop()

# ==========================
# SIDEBAR (After Login)
# ==========================
with st.sidebar:
    st.success("✅ Authenticated")
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    st.markdown("### Scanner Controls")
    auto_refresh = st.checkbox("Auto-refresh every 5 min", value=False)
    st.markdown("### Last Run")
    if "last_run" in st.session_state:
        st.caption(st.session_state.last_run)

# ==========================
# MAIN APP
# ==========================
st.title("📊 Heikin Ashi Daily Futures Scanner")
st.caption("End-of-Day | Risk-Aware | 23 Liquid Symbols | **4-Signal Core + Filters**")

# FIXED: Cache-busting + Safe column selection
@st.cache_data(ttl=3600, show_spinner=False)
def cached_scanner():
    return run_scanner()

# Force cache clear on button press
if st.button("🔄 Run Scanner Now", type="primary") or ("last_run" not in st.session_state):
    # Clear cache explicitly
    cached_scanner.clear()
    with st.spinner("Running scanner..."):
        report = cached_scanner()
    st.session_state.last_run = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.rerun()
else:
    report = cached_scanner()

# Display results
if report.empty:
    st.info("No signals generated today.")
else:
    st.success(f"Scan complete – {len(report)} results found")

    # === PINNED CORE COLUMNS (Decision columns)
    pinned_cols = ["Ticker", "Final_Score", "Final_Verdict", "Breakout", "HA", "Fut_Bias"]

    # === DATAFRAME WITH PINNED COLUMNS ===
    st.dataframe(
        report,
        width="stretch",
        hide_index=True,
        column_config={
            col: st.column_config.Column(pinned=True)
            for col in pinned_cols
            if col in report.columns
        }
    )

    # === UPDATED METRICS (New scoring: STRONG ≥5, WEAK 1-4.9) ===
    col1, col2, col3, col4, col5 = st.columns(5)
    
    strong_buy = (report["Final_Score"] >= 5).sum()
    strong_sell = (report["Final_Score"] <= -5).sum()
    weak_buy = ((report["Final_Score"] > 1) & (report["Final_Score"] < 5)).sum()
    weak_sell = ((report["Final_Score"] < -1) & (report["Final_Score"] > -5)).sum()
    neutral = (abs(report["Final_Score"]) <= 1).sum()

    col1.metric("Total", len(report), delta=f"+{len(report)}")
    col2.metric("STRONG BUY", strong_buy, delta=f"+{strong_buy}")
    col3.metric("STRONG SELL", strong_sell, delta=f"+{strong_sell}")
    col4.metric("WEAK", weak_buy + weak_sell, delta=f"+{weak_buy + weak_sell}")
    col5.metric("NEUTRAL", neutral, delta=f"+{neutral}")

    # === TRADE RECOMMENDATIONS ===
    st.markdown("---")
    st.markdown("### 🚀 **TRADE RECOMMENDATIONS**")
    
    strong_df = report[report["Final_Score"] >= 5].copy()
    if not strong_df.empty:
        st.success(f"**{len(strong_df)} STRONG BUY signals** - Execute tomorrow!")
        st.dataframe(
            strong_df[["Ticker", "Final_Score", "Breakout", "HA", "Fut_Bias", "Vol_Ratio", 
                      "Breakout_Conf", "Compression", "ADX_Trend"]],
            width="stretch",
            height=200
        )
    else:
        st.warning("No STRONG BUY signals today")

    # === FALSE BREAKOUT FILTER SUMMARY ===
    st.markdown("### 🛡️ **False Breakout Filters**")
    col1, col2, col3 = st.columns(3)
    
    confirmed = len(report[(report["Final_Score"] >= 5) & (report["Breakout_Conf"] == "YES")])
    compressed = len(report[(report["Final_Score"] >= 5) & (report["Compression"] == "YES")])
    strong_trend = len(report[(report["Final_Score"] >= 5) & (report["ADX_Trend"] == "STRONG")])
    
    col1.metric("2-Bar Confirmed", confirmed)
    col2.metric("Compression Setup", compressed)
    col3.metric("ADX Strong", strong_trend)

    # === POSITION SIZING GUIDE ===
    st.markdown("### 💰 **Position Sizing (₹2.5L Capital, 1% Risk)**")
    if not strong_df.empty:
        for idx, row in strong_df.head(3).iterrows():
            ticker = row["Ticker"]
            vol_ratio = row["Vol_Ratio"]
            score = row["Final_Score"]
            
            size = "HIGH" if vol_ratio > 2 else "MEDIUM" if vol_ratio > 1.2 else "LOW"
            st.caption(f"**{ticker}**: Score {score} | Vol {vol_ratio}x | Size: **{size}**")

    # Download
    csv = report.to_csv(index=False).encode()
    st.download_button(
        label="📥 Download Full CSV (23 cols)",
        data=csv,
        file_name=f"HA_Scanner_{date.today()}.csv",
        mime="text/csv"
    )

# Auto-refresh
if auto_refresh:
    st.autorefresh(interval=5*60*1000, key="auto")
