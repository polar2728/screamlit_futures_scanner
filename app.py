# app.py
import streamlit as st
import pandas as pd
import yaml
import streamlit_authenticator as stauth
from pathlib import Path

st.set_page_config(page_title="HA Daily Scanner", layout="wide")

# ==========================
# AUTH
# ==========================
with open("users.yaml") as f:
    config = yaml.safe_load(f)

authenticator = stauth.Authenticate(
    config["credentials"],
    cookie_name="ha_scanner",
    key="secure_key_123",
    cookie_expiry_days=7
)

name, auth_status, username = authenticator.login("Login", "main")

if auth_status is False:
    st.error("Invalid username or password")
elif auth_status is None:
    st.warning("Please login")
else:
    authenticator.logout("Logout", "sidebar")

    st.title("📊 Heikin Ashi Daily Scanner")
    st.caption("Spot-based | End-of-Day | Risk-aware")

    # ==========================
    # LOAD LATEST REPORT
    # ==========================
    files = sorted(Path(".").glob("HA_Daily_Scanner_*.csv"), reverse=True)

    if not files:
        st.warning("No scan reports found. Run scanner.py first.")
        st.stop()

    df = pd.read_csv(files[0])

    st.success(f"Loaded report: {files[0].name}")

    # ==========================
    # FILTERS
    # ==========================
    verdict_filter = st.multiselect(
        "Filter Verdict",
        df["Verdict"].unique(),
        default=["STRONG BUY", "STRONG SELL"]
    )

    min_conf = st.slider("Minimum Confidence %", 0, 100, 40)

    filtered = df[
        (df["Verdict"].isin(verdict_filter)) &
        (df["Confidence_%"] >= min_conf)
    ]

    # ==========================
    # DISPLAY
    # ==========================
    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### 🔎 Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Signals", len(filtered))
    col2.metric("Strong Buys", (filtered["Verdict"] == "STRONG BUY").sum())
    col3.metric("Strong Sells", (filtered["Verdict"] == "STRONG SELL").sum())
