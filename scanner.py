import pandas as pd
import numpy as np
import requests
import zipfile
import yfinance as yf
from io import BytesIO
from datetime import date, timedelta
import time
import talib

# ==========================================================
# CONFIG
# ==========================================================
LOOKBACK_DAYS = "9mo"
MAX_RETRIES = 3
MIN_ATR_PCT = 0.6
INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY"}
USE_ALL_FNO = True
BONUS_VOL_RATIO = True  # +1 if Vol_Ratio > 2 on breakout day

CORE_TICKERS = [
    "NIFTY", "BANKNIFTY", "RELIANCE", "HDFCBANK", "ICICIBANK", "AXISBANK", "SBIN",
    "INFY", "TCS", "ITC", "HINDUNILVR", "BHARTIARTL", "LT", "KOTAKBANK",
    "TATASTEEL", "COALINDIA", "ASIANPAINT", "MARUTI", "DRREDDY",
    "TATAPOWER", "INDIGO", "ULTRACEMCO", "ONGC", "BAJFINANCE"
]

# ==========================================================
# SAFE YFINANCE DOWNLOAD
# ==========================================================
def safe_yf_download(symbol):
    for _ in range(MAX_RETRIES):
        try:
            df = yf.download(symbol, period=LOOKBACK_DAYS, interval="1d",
                             auto_adjust=True, progress=False, threads=False)
            if not df.empty and len(df) > 50:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df.reset_index(drop=True)
        except Exception:
            time.sleep(0.5)
    return pd.DataFrame()

# ==========================================================
# HEIKIN ASHI
# ==========================================================
def heikin_ashi(df):
    if df.empty or len(df) < 2:
        return pd.DataFrame()

    ha = df.copy()
    ha["HA_Close"] = ha[["Open", "High", "Low", "Close"]].mean(axis=1)
    ha_open = np.full(len(ha), np.nan)
    ha_open[0] = (ha["Open"].iloc[0] + ha["Close"].iloc[0]) / 2
    for i in range(1, len(ha)):
        ha_open[i] = (ha_open[i-1] + ha["HA_Close"].iloc[i-1]) / 2
    ha["HA_Open"] = ha_open
    ha["HA_High"] = ha[["High", "HA_Open", "HA_Close"]].max(axis=1)
    ha["HA_Low"] = ha[["Low", "HA_Open", "HA_Close"]].min(axis=1)
    return ha

# ==========================================================
# F&O HELPERS
# ==========================================================
def get_nse_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com"})
    try:
        s.get("https://www.nseindia.com", timeout=5)
    except:
        pass
    return s

def fetch_ban_list():
    url = "https://nsearchives.nseindia.com/content/fo/fo_secban.csv"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if r.status_code != 200:
            return set()
        text = r.text.strip()
        if "Securities in Ban" not in text:
            return set()
        header = text.splitlines()[0]
        ban_part = header.split(":", 1)[1].strip() if ':' in header else ""
        banned = set()
        for item in ban_part.split():
            if ',' in item:
                sym = item.split(",", 1)[1].strip().upper()
                if sym.isalpha():
                    banned.add(sym)
        print(f"🚫 Banned symbols: {len(banned)}")
        return banned
    except Exception:
        print("⚠️ Ban list unavailable")
        return set()

def download_fo():
    s = get_nse_session()
    for i in range(7):
        d = date.today() - timedelta(days=i)
        url = f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv.zip"
        try:
            r = s.get(url, timeout=10)
            if r.status_code == 200:
                with zipfile.ZipFile(BytesIO(r.content)) as z:
                    print(f"✅ F&O bhavcopy loaded for {d}")
                    return pd.read_csv(z.open(z.namelist()[0]))
        except Exception:
            continue
    print("⚠️ No F&O bhavcopy found")
    return pd.DataFrame()

# ==========================================================
# BUILD TICKER UNIVERSE
# ==========================================================
def build_ticker_universe():
    print("⏳ Building ticker universe...")
    if not USE_ALL_FNO:
        tickers = sorted(set(CORE_TICKERS))
    else:
        fo = download_fo()
        if fo.empty or "FinInstrmTp" not in fo.columns or "TckrSymb" not in fo.columns:
            print("⚠️ Fallback to core tickers")
            tickers = sorted(set(CORE_TICKERS))
        else:
            banned = fetch_ban_list()
            stocks = fo[fo["FinInstrmTp"] == "STF"]["TckrSymb"].str.upper().unique()
            valid = set(stocks) - banned
            tickers = sorted(valid | INDEX_SYMBOLS)
            print(f"✅ Loaded {len(tickers)} F&O symbols")
    symbol_map = {
        t: "^NSEI" if t == "NIFTY" else "^NSEBANK" if t == "BANKNIFTY" else f"{t}.NS"
        for t in tickers
    }
    return symbol_map

# ==========================================================
# ANALYZE CASH (sequential, stable)
# ==========================================================
def analyze_cash(name, symbol, market_regime):
    df = safe_yf_download(symbol)
    if df.empty or len(df) < 50:
        return None

    # Safe scalar extraction
    try:
        price = df["Close"].iloc[-1].item()
        dh = df["High"].rolling(20).max().shift(1).iloc[-1].item()
        dl = df["Low"].rolling(20).min().shift(1).iloc[-1].item()
    except Exception:
        return None

    if pd.isna(price) or pd.isna(dh) or pd.isna(dl):
        return None

    breakout = "LONG" if price > dh else "SHORT" if price < dl else "NONE"

    ha = heikin_ashi(df)
    if ha.empty or len(ha) < 50:
        return None

    try:
        ema20 = ha["HA_Close"].ewm(span=20, adjust=False).mean().iloc[-1].item()
        ema50 = ha["HA_Close"].ewm(span=50, adjust=False).mean().iloc[-1].item()
        ha_close_today = ha["HA_Close"].iloc[-1].item()
        ha_open_today = ha["HA_Open"].iloc[-1].item()
        ha_close_prev = ha["HA_Close"].iloc[-2].item()
        ha_open_prev = ha["HA_Open"].iloc[-2].item()
    except Exception:
        return None

    trend = "UP" if ema20 > ema50 * 1.02 else "DOWN" if ema20 < ema50 * 0.98 else "SIDE"
    # === FULL DOJ I HANDLING ===
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1].item()
    doji_threshold = atr * 0.1

    ha_body_today = abs(ha_close_today - ha_open_today)
    if ha_body_today <= doji_threshold:
        ha_today = "DOJI"
        ha_contribution = 0
    else:
        ha_today = "BULL" if ha_close_today > ha_open_today else "BEAR"
        ha_contribution = 1 if ha_today == "BULL" else -1

    ha_body_prev = abs(ha_close_prev - ha_open_prev)
    if ha_body_prev <= doji_threshold:
        prev_ha = "DOJI"
    else:
        prev_ha = "BULL" if ha_close_prev > ha_open_prev else "BEAR"
    # ==============================

    # Large body bonus (only if not Doji)
    if ha_body_today > atr * 0.5:
        score_add_body = 0.5 if ha_today == "BULL" else -0.5
    else:
        score_add_body = 0

    # RSI
    delta = ha["HA_Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    rsi = 100 - (100 / (1 + (gain / (loss + 1e-9)))).iloc[-1].item()

    # ATR%
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1].item()
    atr_pct = (atr / price) * 100

    # ADX
    adx = talib.ADX(df["High"].values, df["Low"].values, df["Close"].values, timeperiod=14)[-1]

    compression = "YES" if atr_pct < 0.8 and adx < 20 else "NO"

    # Volume ratio
    vol_ratio = df["Volume"].iloc[-1].item() / df["Volume"].rolling(20).mean().iloc[-1].item()

    # Score
    score = 0
    score += 3 if breakout == "LONG" else -3 if breakout == "SHORT" else 0
    score += 1 if ha_today == "BULL" else -1

    # NEW: Large HA body contribution (+0.5 for strong momentum candle)
    # ha_body = abs(ha["HA_Close"].iloc[-1].item() - ha["HA_Open"].iloc[-1].item())
    # if ha_body > atr * 0.5:
    #     score += 0.5 if ha_today == "BULL" else -0.5

    score += ha_contribution
    score += score_add_body  # Large body bonus

    if BONUS_VOL_RATIO and vol_ratio > 2 and breakout != "NONE":
        score += 1
    if atr_pct < MIN_ATR_PCT:
        score *= 0.5

    return {
        "Ticker": name,
        "Final_Score": round(score, 2),
        "Final_Verdict": "STRONG BUY" if score >= 5 else "STRONG SELL" if score <= -5 else
                         "WEAK BUY" if score > 1 else "WEAK SELL" if score < -1 else "NEUTRAL",
        "Breakout": breakout,
        "HA": ha_today,
        "Prev_HA": prev_ha,
        "Trend": trend,
        "RSI": round(rsi, 1),
        "ATR%": round(atr_pct, 2),
        "Compression": compression,
        "ADX": round(adx, 1),
        "Vol_Ratio": round(vol_ratio, 2),
        "Price": round(price, 2),
        "Market_Regime": market_regime
    }

# ==========================================================
# RUN SCANNER (single-threaded)
# ==========================================================
def run_scanner():
    symbol_map = build_ticker_universe()
    print(f"🚀 Scanner started for {len(symbol_map)} symbols (single-threaded)...")

    # Market regime (single Nifty download)
    nifty_df = safe_yf_download("^NSEI")
    market_regime = "NEUTRAL"
    if not nifty_df.empty:
        ema20 = nifty_df["Close"].ewm(span=20, adjust=False).mean().iloc[-1].item()
        ema50 = nifty_df["Close"].ewm(span=50, adjust=False).mean().iloc[-1].item()
        market_regime = "RISK_ON" if ema20 > ema50 else "RISK_OFF"

    results = []
    total = len(symbol_map)
    for i, (name, sym) in enumerate(symbol_map.items(), 1):
        # print(f"Processing {i}/{total}: {name}")
        result = analyze_cash(name, sym, market_regime)
        if result:
            results.append(result)

    cash_df = pd.DataFrame(results)
    fut_df = run_futures(list(symbol_map.keys()))

    df = cash_df.merge(fut_df, on="Ticker", how="left")
    return df.sort_values("Final_Score", ascending=False).reset_index(drop=True)

# ==========================================================
# FUTURES
# ==========================================================
def futures_signal(row):
    pc = row["ClsPric"] - row["OpnPric"]
    oi = row["ChngInOpnIntrst"]
    if pc > 0 and oi > 0: return "Long Buildup"
    if pc < 0 and oi > 0: return "Short Buildup"
    if pc > 0 and oi < 0: return "Short Covering"
    return "Long Unwinding"

def run_futures(tickers):
    fo = download_fo()
    if fo.empty:
        return pd.DataFrame()
    out = []
    for sym in tickers:
        f = fo[(fo["TckrSymb"] == sym) & (fo["FinInstrmTp"].isin(["STF", "IDF"]))].copy()
        if f.empty:
            continue
        f["XpryDt"] = pd.to_datetime(f["XpryDt"])
        f = f.sort_values("XpryDt").head(2)
        row = {"Ticker": sym}
        for i, (_, r) in enumerate(f.iterrows(), 1):
            row.update({
                f"F{i}_Expiry": r["XpryDt"].date(),
                f"F{i}_Close": r["ClsPric"],
                f"F{i}_OI_Change": r["ChngInOpnIntrst"],
                f"F{i}_OI_%": round(r["ChngInOpnIntrst"] / r["OpnIntrst"] * 100, 2) if r["OpnIntrst"] > 0 else 0,
                f"F{i}_Signal": futures_signal(r)
            })
        out.append(row)
    return pd.DataFrame(out)

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    df = run_scanner()
    print("\nTop 15 signals:")
    print(df.head(15).to_string(index=False))
    csv_file = f"scanner_output_{date.today().strftime('%Y%m%d')}.csv"
    df.to_csv(csv_file, index=False)
    print(f"\n✅ Full report saved: {csv_file} ({len(df)} rows)")