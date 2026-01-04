import streamlit as st
import bcrypt
import pandas as pd
from datetime import datetime, date
import time
import scanner
from scanner import run_scanner, run_futures

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
    use_all_fno = st.checkbox("Scan ALL F&O (~200+)", value=False)
    auto_refresh = st.checkbox("Auto-refresh every 5 min", value=False)
    st.markdown("### Last Run")
    if "last_run" in st.session_state:
        st.caption(st.session_state.last_run)

# ==========================
# CACHED FUTURES
# ==========================
@st.cache_data(ttl=86400)
def cached_run_futures(_tickers):
    return run_futures(_tickers)

# ==========================
# SCANNER WITH CACHED FUTURES + RATE LIMIT HANDLING
# ==========================
@st.cache_data(ttl=3600, show_spinner=False)
def run_scanner_optimized(_use_all_fno: bool):
    scanner.USE_ALL_FNO = _use_all_fno
    symbol_map = scanner.build_ticker_universe()

    # Safe Nifty download with fallback
    nifty_df = scanner.safe_yf_download("^NSEI")
    market_regime = "NEUTRAL"
    nifty_return = None
    if not nifty_df.empty:
        try:
            ema20 = nifty_df["Close"].ewm(span=20, adjust=False).mean().iloc[-1].item()
            ema50 = nifty_df["Close"].ewm(span=50, adjust=False).mean().iloc[-1].item()
            market_regime = "RISK_ON" if ema20 > ema50 else "RISK_OFF"
            if len(nifty_df) > 60:
                nifty_return = (nifty_df["Close"].iloc[-1] / nifty_df["Close"].iloc[-60]) - 1
        except:
            pass
    else:
        st.warning("⚠️ Nifty data rate-limited — using NEUTRAL regime (try again later)")

    results = []
    for name, sym in symbol_map.items():
        result = scanner.analyze_cash(name, sym, market_regime, nifty_return)
        if result:
            results.append(result)

    cash_df = pd.DataFrame(results)
    tickers_list = list(symbol_map.keys())
    fut_df = cached_run_futures(tickers_list)
    df = cash_df.merge(fut_df, on="Ticker", how="left")

    df = scanner.add_futures_scoring(df)

    return df.sort_values("Final_Score", ascending=False).reset_index(drop=True)

# ==========================
# MAIN APP
# ==========================
st.title("📊 Donchian + Reversal Scanner")
st.caption("Early reversal detection • Futures confirmation • Conviction ranked")

with st.expander("📘 Interpretation Guide", expanded=False):
    st.markdown("""
        **How to Read This Scanner**:
        - **Score** = Breakout + HA momentum + Reversal bonus + Comp + Volume + Futures
        - **STRONG BUY** ≥6 | **WEAK BUY** >2

        **Key Features**:
        - Donchain Channel Breakout
        - Heikin Ashi BULL / BEAR = clear momentum And **DOJI** = indecision 
        - Reversal Bonus: +1.5 if strong candle follows Doji or opposite → early reversal
        - Futures Confirmation: Long Buildup adds ~1–1.5 → smart money alignment
        - Comp = YES → coiled spring (high-probability breakout)

        **Relative Strength vs Nifty (RS_vs_Nifty)**:
        - STRONG = stock outperformed Nifty by >20% over 3 months → true leadership
        - GOOD = outperforming Nifty
        - WEAK = lagging Nifty

        **Range Position (60-day)**:
        - UPPER = breakout from top 25% of range → strongest
        - MIDDLE = middle of range
        - LOWER = bottom 40% → weakest

        **Strength Tier** (S/A/B/C):
        - Composite rank based on breakout, HA, compression, RS, range position, volume
        - **Elite** — top-tier (rare, highest probability)
        - **High** — strong conviction
        - **Good** — solid
        - **Avg** — average

        **ATR%** (Average True Range %):
        - < 0.8% = low volatility (potential compression)

        **ADX**:
        - > 25 = strong trend and < 20 = sideways

        **Compression = YES** 🔥:
        - Low volatility + weak trend → "coiled spring" setup

        **Bold rows** = Highest conviction:
        - Strong verdict + (Compression YES or ADX > 25) + Long Buildup/Short Covering in futures
    """)

# ==========================
# RUN + CACHE CONTROL
# ==========================
col1, col2 = st.columns([1, 4])
with col1:
    run_now = st.button("🔄 Run Scanner Now", type="primary")
with col2:
    clear_cache = st.button("🗑️ Clear Cache (Force Fresh Data)")

if clear_cache:
    st.cache_data.clear()
    st.success("Cache cleared — next run will fetch fresh data")
    st.rerun()

if run_now or ("last_run" not in st.session_state):
    run_scanner_optimized.clear()
    mode = "ALL F&O (~200+)" if use_all_fno else "Core Stocks"
    start_time = time.time()
    with st.spinner(f"Running scanner ({mode})..."):
        report = run_scanner_optimized(use_all_fno)
    runtime = time.time() - start_time
    st.session_state.last_run = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.success(f"Scan complete in {runtime:.1f}s — {len(report)} symbols")
else:
    report = run_scanner_optimized(use_all_fno)

# ==========================
# HELPERS (updated for renamed columns)
# ==========================
def build_trade_thesis(row):
    thesis = []

    thesis.append(
        f"**Market Regime:** {row['Regime']} — "
        f"{'supports' if row['Regime']=='RISK_ON' else 'does not favor'} long trades."
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

    if row.get("Futures_Score", 0) > 0.5:
        thesis.append("**Reversal pattern** detected (strong candle after Doji/opposite).")

    thesis.append(f"Trend: **{row['Trend']}** | ADX: **{row['ADX']:.1f}**")

    if pd.notna(row.get("F1_Signal")):
        thesis.append(f"**Futures:** {row['F1_Signal']} (adds to score)")

    thesis.append(
        f"**Conviction (ST):** {row['ST']} | Score: {row['Score']:.1f} → **{row['Verdict']}**"
    )

    recommendation = (
        "✅ **High-conviction** — consider entry (ST = Elite/High + futures support)."
        if row["Verdict"] in ["STRONG BUY", "WEAK BUY"] and row["ST"] in ["Elite", "High"]
        else "✅ **Consider entry** with tight risk."
        if row["Verdict"] in ["STRONG BUY", "WEAK BUY"]
        else "⚠️ **Avoid / Reduce**"
        if row["Verdict"] in ["WEAK SELL", "STRONG SELL"]
        else "⏸️ **Wait for confirmation**"
    )

    return "\n\n".join(thesis), recommendation

# ==========================
# DISPLAY RESULTS
# ==========================
if report.empty:
    st.info("No data — run the scanner.")
else:
    mode = "ALL F&O (~200+)" if use_all_fno else "Core Stocks"

    # === MOVE RENAMING HERE (before metrics) ===
    report = report.rename(columns={
        "Strength_Tier": "ST",
        "Compression": "Comp",
        "Final_Score": "Score",
        "Final_Verdict": "Verdict",
        "Market_Regime": "Regime"
    })

    st_map = {"S": "Elite", "A": "High", "B": "Good", "C": "Avg"}
    report["ST"] = report["ST"].map(st_map)

    # Search bar
    search_term = st.text_input("🔍 Search Ticker", "")
    display_df = report.copy()
    if search_term:
        display_df = display_df[display_df["Ticker"].str.contains(search_term.upper(), case=False)]

    # Formatting
    format_dict = {}
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            if col in ["RSI", "ADX"]:
                format_dict[col] = "{:.1f}"
            elif "OI_%" in col:
                format_dict[col] = "{:.2f}"
            elif col in ["Score", "ATR%", "Vol_Ratio", "Futures_Score"] or "Close" in col:
                format_dict[col] = "{:.2f}"
            elif "OI_Change" in col:
                format_dict[col] = "{:,.0f}"
            else:
                format_dict[col] = "{:.2f}"

    def highlight_row(row):
        is_signal = row["Verdict"] in ["STRONG BUY", "WEAK BUY", "STRONG SELL", "WEAK SELL"]
        good_futures = pd.notna(row.get("F1_Signal")) and row["F1_Signal"] in ["Long Buildup", "Short Covering"]
        comp_or_trend = (row["Comp"] == "YES") or (row["ADX"] > 25)

        if is_signal and good_futures and comp_or_trend:
            return ['font-weight: bold'] * len(row)
        return [''] * len(row)

    styled = display_df.style.format(format_dict).apply(highlight_row, axis=1)

    pinned = ["Ticker", "Verdict", "Score", "ST", "Breakout", "Comp"]

    st.dataframe(
        styled,
        hide_index=True,
        column_config={c: st.column_config.Column(pinned=True) for c in pinned if c in display_df.columns}
    )

    # Metrics (now safe — uses renamed columns)
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