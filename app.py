import streamlit as st
import bcrypt
import pandas as pd
from datetime import datetime, date
from scanner import run_scanner

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="HA Daily Scanner",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
# SIDEBAR
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
# HELPERS
# ==========================
def reco_icon(verdict):
    return {
        "STRONG BUY": "🟢🚀",
        "WEAK BUY": "🟡📈",
        "NEUTRAL": "⚪⏸️",
        "WEAK SELL": "🟡📉",
        "STRONG SELL": "🔴💣"
    }.get(verdict, "❓")

def build_trade_thesis(row):
    thesis = []

    thesis.append(
        f"**Market Regime:** {row['Market_Regime']} — "
        f"{'supports' if row['Market_Regime']=='RISK_ON' else 'does not favor'} long trades."
    )

    if row["Breakout"] == "LONG":
        thesis.append("Price has broken above recent resistance (Donchian breakout).")
    elif row["Breakout"] == "SHORT":
        thesis.append("Price has broken below recent support.")

    thesis.append(f"Heikin Ashi candles are **{row['HA']}**, indicating short-term momentum.")

    thesis.append(
        f"Trend structure is **{row['Trend']}** with ADX showing "
        f"**{row['ADX']}** trend strength."
    )

    if pd.notna(row.get("Fut_Bias")):
        thesis.append(f"Near-month futures indicate **{row['Fut_Bias']}**.")

    if pd.notna(row.get("Next_Fut_Bias")):
        thesis.append(f"Next-month futures show **{row['Next_Fut_Bias']}**, reflecting forward positioning.")

    thesis.append(
        f"**Final Score:** {row['Final_Score']} → **{row['Final_Verdict']}**"
    )

    recommendation = (
        "✅ **Hold / Add on dips** with defined risk."
        if row["Final_Verdict"] in ["STRONG BUY", "WEAK BUY"]
        else "⚠️ **Avoid fresh longs / Reduce exposure**"
        if row["Final_Verdict"] in ["WEAK SELL", "STRONG SELL"]
        else "⏸️ **Wait for confirmation**"
    )

    return "\n\n".join(thesis), recommendation

# ==========================
# MAIN APP
# ==========================
st.title("📊 Heikin Ashi Daily Futures Scanner")
st.caption("End-of-Day | Risk-Aware | Cash + Futures Conviction Engine")

@st.cache_data(ttl=3600, show_spinner=False)
def cached_scanner():
    return run_scanner()

# Run scanner
if st.button("🔄 Run Scanner Now", type="primary") or ("last_run" not in st.session_state):
    cached_scanner.clear()
    with st.spinner("Running scanner..."):
        report = cached_scanner()
    st.session_state.last_run = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.rerun()
else:
    report = cached_scanner()

# ==========================
# DISPLAY RESULTS
# ==========================
if report.empty:
    st.info("No signals generated today.")
else:
    report.insert(1, "Reco", report["Final_Verdict"].apply(reco_icon))

    st.success(f"Scan complete – {len(report)} symbols analyzed")

    pinned_cols = ["Ticker", "Reco", "Final_Score", "Final_Verdict"]

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

    # ==========================
    # METRICS
    # ==========================
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total", len(report))
    col2.metric("STRONG BUY", (report["Final_Score"] >= 5).sum())
    col3.metric("STRONG SELL", (report["Final_Score"] <= -5).sum())
    col4.metric("WEAK", ((report["Final_Score"].abs() > 1) & (report["Final_Score"].abs() < 5)).sum())
    col5.metric("NEUTRAL", (report["Final_Score"].abs() <= 1).sum())

    # ==========================
    # TRADE EXPLAINER
    # ==========================
    st.markdown("---")
    st.markdown("## 🧠 Trade Explanation Engine")

    selected = st.selectbox(
        "Select a ticker to explain:",
        report["Ticker"].tolist()
    )

    row = report[report["Ticker"] == selected].iloc[0]
    thesis, recommendation = build_trade_thesis(row)

    st.subheader(f"{row['Reco']} {row['Ticker']} — {row['Final_Verdict']}")
    st.markdown(thesis)
    st.success(recommendation)

    # ==========================
    # DOWNLOAD
    # ==========================
    csv = report.to_csv(index=False).encode()
    st.download_button(
        "📥 Download Full CSV",
        data=csv,
        file_name=f"HA_Scanner_{date.today()}.csv",
        mime="text/csv"
    )

# ==========================
# AUTO REFRESH
# ==========================
if auto_refresh:
    st.autorefresh(interval=5 * 60 * 1000, key="auto")
