import streamlit as st
import bcrypt
import pandas as pd
from datetime import datetime, date
import scanner
from scanner import run_scanner

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Donchian Breakout Scanner",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# LOGIN
# ==========================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Login to Scanner")
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
    use_all_fno = st.checkbox("Scan ALL F&O (~200+)", value=True)
    auto_refresh = st.checkbox("Auto-refresh every 5 min", value=False)
    st.markdown("### Last Run")
    if "last_run" in st.session_state:
        st.caption(st.session_state.last_run)

# ==========================
# HELPERS
# ==========================
def build_trade_thesis(row):
    thesis = []

    thesis.append(
        f"**Market Regime:** {row['Market_Regime']} — "
        f"{'supports' if row['Market_Regime']=='RISK_ON' else 'does not favor'} long trades."
    )

    if row["Breakout"] == "LONG":
        thesis.append("**Donchian breakout bullish** (above 20-day high).")
    elif row["Breakout"] == "SHORT":
        thesis.append("**Donchian breakout bearish** (below 20-day low).")

    ha_desc = row['HA']
    if ha_desc == "DOJI":
        thesis.append("Heikin Ashi today: **DOJI** → indecision.")
    else:
        thesis.append(f"Heikin Ashi today: **{ha_desc}** → momentum.")

    if row.get("Futures_Score", 0) != 0:
        thesis.append("**Reversal pattern** detected (strong candle after Doji/opposite).")

    thesis.append(f"Trend: **{row['Trend']}** | ADX: **{row['ADX']:.1f}**")

    if pd.notna(row.get("F1_Signal")):
        thesis.append(f"**Futures:** {row['F1_Signal']} (adds to score)")

    thesis.append(
        f"**Conviction (ST):** {row['ST']} | Score: {row['Final_Score']} → **{row['Final_Verdict']}**"
    )

    recommendation = (
        "✅ **High-conviction** — consider entry (ST = Elite/High + futures support)."
        if row["Final_Verdict"] in ["STRONG BUY", "WEAK BUY"] and row["ST"] in ["Elite", "High"]
        else "✅ **Consider entry** with tight risk."
        if row["Final_Verdict"] in ["STRONG BUY", "WEAK BUY"]
        else "⚠️ **Avoid / Reduce**"
        if row["Final_Verdict"] in ["WEAK SELL", "STRONG SELL"]
        else "⏸️ **Wait for confirmation**"
    )

    return "\n\n".join(thesis), recommendation

# ==========================
# MAIN APP
# ==========================
st.title("📊 Donchian + Reversal Scanner")
st.caption("Early reversal detection • Futures confirmation • Conviction ranked")

# ==========================
# CONCISE GUIDE (previous style + new features)
# ==========================
with st.expander("📘 Interpretation Guide", expanded=False):
    st.markdown("""
    **Core Logic**:
    - **Score** = Breakout + HA momentum + Reversal bonus + Compression + Volume + Futures
    - **STRONG BUY** ≥6 | **WEAK BUY** >2 (higher = stronger)

    **New Features**:
    - **Reversal Bonus**: +1.5 if strong candle follows Doji or opposite candle → early reversal flag
    - **Futures Confirmation**: Long Buildup/Short Covering adds ~1–1.5 to score
    - **Comp = YES** → coiled spring (high probability breakout)

    **Conviction (ST)**:
    - **Elite** = top-tier setup (rare, highest probability)
    - **High** = strong conviction
    - **Good** = solid
    - **Avg** = average

    **Focus on**:
    - Elite/High ST
    - Futures = Long Buildup
    - Comp = YES
    - Reversal pattern + high volume

    Bold rows = highest conviction (strong verdict + futures + compression/trend).
    """)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_scanner(_use_all_fno: bool):
    scanner.USE_ALL_FNO = _use_all_fno
    return run_scanner()

run_now = st.button("🔄 Run Scanner Now", type="primary")

if run_now or ("last_run" not in st.session_state):
    cached_scanner.clear()
    mode = "ALL F&O (~200+)" if use_all_fno else "Core Stocks"
    with st.spinner(f"Running scanner ({mode})..."):
        report = cached_scanner(use_all_fno)
    st.session_state.last_run = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.rerun()
else:
    report = cached_scanner(use_all_fno)

# ==========================
# DISPLAY RESULTS
# ==========================
if report.empty:
    st.info("No data — run the scanner.")
else:
    mode = "ALL F&O (~200+)" if use_all_fno else "Core Stocks"
    st.success(f"Scan complete — {len(report)} symbols ({mode})")

    # Rename for space
    report = report.rename(columns={
        "Strength_Tier": "ST",
        "Compression": "Comp",
        "Final_Score": "Score",
        "Final_Verdict": "Verdict",
        "Market_Regime": "Regime"
    })

    # Map ST values for readability
    st_map = {"S": "Elite", "A": "High", "B": "Good", "C": "Avg"}
    report["ST"] = report["ST"].map(st_map)

    # Decimal formatting (concise)
    format_dict = {
        "Score": "{:.1f}",
        "RSI": "{:.0f}",
        "ATR%": "{:.2f}",
        "Vol_Ratio": "{:.1f}",
        "ADX": "{:.0f}",
        "Price": "{:,.0f}",
        "Futures_Score": "{:.1f}",
    }

    def highlight_row(row):
        is_signal = row["Verdict"] in ["STRONG BUY", "WEAK BUY", "STRONG SELL", "WEAK SELL"]
        good_futures = pd.notna(row.get("F1_Signal")) and row["F1_Signal"] in ["Long Buildup", "Short Covering"]
        comp_or_trend = (row["Comp"] == "YES") or (row["ADX"] > 25)

        if is_signal and good_futures and comp_or_trend:
            return ['font-weight: bold'] * len(row)
        return [''] * len(row)

    styled = report.style.format(format_dict).apply(highlight_row, axis=1)

    # Pinned + space-saving columns
    pinned = ["Ticker", "Verdict", "Score", "ST", "Breakout", "HA", "Comp"]

    st.dataframe(
        styled,
        hide_index=True,
        column_config={c: st.column_config.Column(pinned=True) for c in pinned if c in report.columns}
    )

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Symbols", len(report))
    col2.metric("STRONG BUY", (report["Score"] >= 6).sum())
    col3.metric("Elite ST", (report["ST"] == "Elite").sum())
    col4.metric("Comp = YES", (report["Comp"] == "YES").sum())

    # Explainer
    st.markdown("---")
    st.markdown("## 🧠 Trade Explainer")

    selected = st.selectbox("Select ticker:", report["Ticker"])
    row = report[report["Ticker"] == selected].iloc[0]
    thesis, reco = build_trade_thesis(row)

    st.subheader(f"{row['Verdict']} | {row['Ticker']} (ST: {row['ST']})")
    st.markdown(thesis)
    st.success(reco)

    # Download
    csv = report.to_csv(index=False).encode()
    st.download_button(
        "📥 Download CSV",
        data=csv,
        file_name=f"scanner_{mode.replace(' ', '_')}_{date.today()}.csv",
        mime="text/csv"
    )

if auto_refresh:
    st.autorefresh(interval=5*60*1000, key="auto")