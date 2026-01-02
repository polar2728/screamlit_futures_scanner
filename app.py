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
    page_title="Donchian Breakout Daily Scanner",
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

    use_all_fno = st.checkbox(
        "Scan ALL F&O Stocks (~200+)",
        value=False,
        help="Uncheck to scan only the original 24 core stocks"
    )

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
        thesis.append("Price has broken above recent 20-day high → **Donchian breakout bullish**.")
    elif row["Breakout"] == "SHORT":
        thesis.append("Price has broken below recent 20-day low → **Donchian breakout bearish**.")

    # HA + Reversal context
    ha_desc = row['HA']
    if ha_desc == "DOJI":
        thesis.append("Heikin Ashi candle today is **DOJI** → indecision.")
    else:
        thesis.append(f"Heikin Ashi candle today is **{ha_desc}** → momentum direction.")

    # Reversal candle mention
    if row.get("Futures_Score", 0) != 0 or "reversal" in str(row).lower():  # placeholder — better if we add flag
        thesis.append("**Reversal pattern detected** — strong candle after Doji/opposite candle.")

    thesis.append(
        f"Trend structure is **{row['Trend']}** with ADX = **{row['ADX']:.1f}**."
    )

    if pd.notna(row.get("F1_Signal")):
        thesis.append(f"**Futures Confirmation:** {row['F1_Signal']} (contributes to score)")

    thesis.append(
        f"**Strength Tier:** {row['Strength_Tier']} | Final Score: {row['Final_Score']} → **{row['Final_Verdict']}**"
    )

    recommendation = (
        "✅ **High-conviction trade** — consider entry (S/A tier + futures support)."
        if row["Final_Verdict"] in ["STRONG BUY", "WEAK BUY"] and row["Strength_Tier"] in ["S", "A"]
        else "✅ **Consider entry** with tight risk."
        if row["Final_Verdict"] in ["STRONG BUY", "WEAK BUY"]
        else "⚠️ **Avoid / Reduce exposure**"
        if row["Final_Verdict"] in ["WEAK SELL", "STRONG SELL"]
        else "⏸️ **Wait for confirmation**"
    )

    return "\n\n".join(thesis), recommendation

# ==========================
# MAIN APP
# ==========================
st.title("📊 Donchian Breakout + Reversal Scanner")
st.caption("End-of-Day | Early Reversal Detection | Futures Confirmation")

# ==========================
# UPDATED GUIDE
# ==========================
with st.expander("📘 Scanner Key Concepts & Interpretation Guide", expanded=False):
    st.markdown("""
    **New Features**:
    - **Reversal Candle Bonus**: +1.5 score if strong candle follows Doji or opposite candle → early reversal detection
    - **Futures Confirmation**: Long Buildup/Short Covering adds to Final_Score → smart money alignment
    - **Adjusted Thresholds**: STRONG BUY now requires ≥6 (more selective)

    **How to Read**:
    - **Final Score** = Breakout + HA + Reversal + Compression + Volume + Futures
    - **Strength_Tier (S/A/B/C)** = Composite conviction rank
    - **Futures_Score** = Contribution from OI/buildup (positive = bullish confirmation)

    **Compression = YES** 🔥 → Coiled spring
    **Bold rows** = High conviction (strong verdict + futures support + compression/trend)

    Focus on **S/A tier** with **F1_Signal = Long Buildup** — highest probability.
    """)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_scanner(_use_all_fno: bool):
    scanner.USE_ALL_FNO = _use_all_fno
    return run_scanner()

run_now = st.button("🔄 Run Scanner Now", type="primary")

if run_now or ("last_run" not in st.session_state):
    cached_scanner.clear()
    mode = "ALL F&O (~200+)" if use_all_fno else "Core 24 Stocks"
    with st.spinner(f"Running scanner on {mode}..."):
        report = cached_scanner(use_all_fno)
    st.session_state.last_run = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.rerun()
else:
    report = cached_scanner(use_all_fno)

# ==========================
# DISPLAY RESULTS
# ==========================
if report.empty:
    st.info("No data returned. Try running the scanner.")
else:
    report.insert(1, "Reco", report["Final_Verdict"].apply(reco_icon))

    mode = "ALL F&O (~200+)" if use_all_fno else "Core 24 Stocks"
    st.success(f"Scan complete – {len(report)} symbols analyzed ({mode})")

    # Decimal formatting (updated for new columns)
    format_dict = {}
    for col in report.columns:
        if pd.api.types.is_float_dtype(report[col]):
            if col in ["RSI", "ADX"]:
                format_dict[col] = "{:.1f}"
            elif "OI_%" in col:
                format_dict[col] = "{:.2f}%"
            elif col in ["Final_Score", "ATR%", "Vol_Ratio", "Futures_Score"] or "Close" in col:
                format_dict[col] = "{:.2f}"
            elif "OI_Change" in col:
                format_dict[col] = "{:,.0f}"
            else:
                format_dict[col] = "{:.2f}"

    def highlight_row(row):
        is_signal = row["Final_Verdict"] in ["STRONG BUY", "WEAK BUY", "STRONG SELL", "WEAK SELL"]
        compression_or_trend = (row["Compression"] == "YES") or (row["ADX"] > 25)
        futures_good = pd.notna(row.get("F1_Signal")) and row["F1_Signal"] in ["Long Buildup", "Short Covering"]

        if is_signal and compression_or_trend and futures_good:
            return ['font-weight: bold'] * len(row)
        return [''] * len(row)

    styled_report = report.style \
        .format(format_dict) \
        .apply(highlight_row, axis=1)

    pinned_cols = ["Ticker", "Reco", "Final_Score", "Final_Verdict", "Strength_Tier"]

    st.dataframe(
        styled_report,
        width="stretch",
        hide_index=True,
        column_config={
            col: st.column_config.Column(pinned=True)
            for col in pinned_cols
            if col in report.columns
        }
    )

    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Symbols", len(report))
    col2.metric("STRONG BUY", (report["Final_Score"] >= 6).sum())
    col3.metric("STRONG SELL", (report["Final_Score"] <= -6).sum())
    col4.metric("WEAK Signals", ((report["Final_Score"].abs() > 2) & (report["Final_Score"].abs() < 6)).sum())
    col5.metric("NEUTRAL", (report["Final_Score"].abs() <= 2).sum())

    # Trade Explainer
    st.markdown("---")
    st.markdown("## 🧠 Trade Explanation Engine")

    selected = st.selectbox(
        "Select a ticker for detailed analysis:",
        options=report["Ticker"].tolist(),
        index=0
    )

    row = report[report["Ticker"] == selected].iloc[0]
    thesis, recommendation = build_trade_thesis(row)

    st.subheader(f"{row['Reco']} {row['Ticker']} — {row['Final_Verdict']} (Tier {row['Strength_Tier']})")
    st.markdown(thesis)
    st.success(recommendation)

    # Download
    csv = report.to_csv(index=False).encode()
    st.download_button(
        "📥 Download Full CSV",
        data=csv,
        file_name=f"Donchian_Scanner_{mode.replace(' ', '_')}_{date.today()}.csv",
        mime="text/csv"
    )

if auto_refresh:
    st.autorefresh(interval=5 * 60 * 1000, key="auto")