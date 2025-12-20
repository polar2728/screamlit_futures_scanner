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
st.caption("End-of-Day | Risk-Aware | 23 Liquid Symbols")

@st.cache_data(ttl=3600, show_spinner=False)  # Cache 1 hour
def cached_scanner():
    return run_scanner()

# Run scanner
if st.button("🔄 Run Scanner Now", type="primary") or ("last_run" not in st.session_state):
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

    # st.dataframe(report, use_container_width=True, hide_index=True)
    st.dataframe(report, width="stretch", hide_index=True)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    strong_buy = (report["Final_Score"] >= 8).sum()
    strong_sell = (report["Final_Score"] <= -8).sum()
    mod_buy = ((report["Final_Score"] >= 4) & (report["Final_Score"] < 8)).sum()
    mod_sell = ((report["Final_Score"] >= -8) & (report["Final_Score"] < -4)).sum()

    col1.metric("Total Signals", len(report))
    col2.metric("Strong Ideas", strong_buy + strong_sell)
    col3.metric("Moderate Ideas", mod_buy + mod_sell)
    col4.metric("Net Edge", f"{(strong_buy + mod_buy) - (strong_sell + mod_sell):+}")


    # Download
    csv = report.to_csv(index=False).encode()
    st.download_button(
        label="📥 Download CSV Report",
        data=csv,
        file_name=f"HA_Scanner_{date.today()}.csv",
        mime="text/csv"
    )

    # assume df is your merged dashboard DataFrame
    pinned_cols = ["Ticker", "Final_Conviction", "Final_Score"]

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

# Auto-refresh (experimental)
if auto_refresh:
    st.autorefresh(interval=5*60*1000, key="auto")