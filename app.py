import pandas as pd
import numpy as np
import requests
import zipfile
import yfinance as yf
from io import BytesIO
from datetime import date, timedelta
import time

# ==========================
# GLOBAL CONFIG
# ==========================

LOOKBACK_DAYS = "9mo"
MAX_RETRIES = 2
MIN_ATR_PCT = 0.6

INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY"}

TICKERS = [
    "NIFTY", "BANKNIFTY", "RELIANCE", "HDFCBANK", "ICICIBANK", "AXISBANK", "SBIN",
    "INFY", "TCS", "ITC", "HINDUNILVR", "BHARTIARTL", "LT", "KOTAKBANK",
    "TATASTEEL", "COALINDIA", "ASIANPAINT", "MARUTI", "DRREDDY",
    "TATAPOWER", "INDIGO", "ULTRACEMCO", "ONGC", "BAJFINANCE"
]

SYMBOL_MAP = {k: f"{k}.NS" for k in TICKERS if k not in INDEX_SYMBOLS}
SYMBOL_MAP.update({"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"})

# ==========================
# SAFE YFINANCE FETCH
# ==========================

def safe_yf_download(symbol):
    for _ in range(MAX_RETRIES):
        try:
            df = yf.download(
                symbol,
                period=LOOKBACK_DAYS,
                interval="1d",
                auto_adjust=True,
                progress=False
            )
            if not df.empty and len(df) > 50:
                return df
        except:
            time.sleep(1)
    return pd.DataFrame()

# ==========================
# HEIKIN ASHI
# ==========================

def heikin_ashi(df):
    df = df.copy()
    df["HA_Close"] = df[["Open", "High", "Low", "Close"]].mean(axis=1)

    ha_open = df[["Open", "Close"]].mean(axis=1)
    ha_open = (ha_open.shift() + df["HA_Close"].shift()).fillna(ha_open) / 2

    df["HA_Open"] = ha_open
    df["HA_High"] = df[["High", "HA_Open", "HA_Close"]].max(axis=1)
    df["HA_Low"] = df[["Low", "HA_Open", "HA_Close"]].min(axis=1)
    return df

# ==========================
# CORE CASH SCANNER
# ==========================

def run_core_scanner():
    results = []

    nifty = safe_yf_download("^NSEI")
    market_regime = "NEUTRAL"
    if not nifty.empty:
        ema20 = nifty["Close"].ewm(span=20).mean().iloc[-1]
        ema50 = nifty["Close"].ewm(span=50).mean().iloc[-1]
        market_regime = "RISK_ON" if float(ema20) > float(ema50) else "RISK_OFF"

    for ticker, symbol in SYMBOL_MAP.items():
        df = safe_yf_download(symbol)
        time.sleep(0.3)

        if df.empty:
            continue

        price = float(df["Close"].iloc[-1])

        don_high = df["High"].rolling(20).max().shift(1).iloc[-1]
        don_low = df["Low"].rolling(20).min().shift(1).iloc[-1]

        breakout = "LONG" if price > don_high else "SHORT" if price < don_low else "NONE"

        ha = heikin_ashi(df)
        ha["EMA20"] = ha["HA_Close"].ewm(span=20).mean()
        ha["EMA50"] = ha["HA_Close"].ewm(span=50).mean()

        delta = ha["HA_Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta).clip(lower=0).rolling(14).mean()
        rs = gain / loss.replace(0, 0.0001)
        ha["RSI"] = 100 - (100 / (1 + rs))

        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs()
        ], axis=1).max(axis=1)
        ha["ATR"] = tr.rolling(14).mean()

        last = ha.iloc[-1]
        atr_pct = (last["ATR"] / price) * 100

        compression = "YES" if atr_pct < 1.2 else "NO"
        vol_ratio = df["Volume"].iloc[-1] / df["Volume"].rolling(20).mean().iloc[-1]

        adx = np.clip(vol_ratio * 10, 10, 40)
        trend = "BULL" if last["EMA20"] > last["EMA50"] else "BEAR"

        score = 0
        score += 3 if breakout == "LONG" else -3 if breakout == "SHORT" else 0
        score += 1 if last["HA_Close"] > last["HA_Open"] else -1
        score += 1 if trend == "BULL" else -1
        score *= 0.5 if atr_pct < MIN_ATR_PCT else 1

        results.append({
            "Ticker": ticker,
            "Final_Score": round(score, 2),
            "Final_Verdict": "STRONG BUY" if score >= 5 else
                             "WEAK BUY" if score > 1 else
                             "STRONG SELL" if score <= -5 else
                             "WEAK SELL" if score < -1 else "NEUTRAL",
            "Breakout": breakout,
            "HA": "BULL" if last["HA_Close"] > last["HA_Open"] else "BEAR",
            "Prev_HA": "BULL" if ha.iloc[-2]["HA_Close"] > ha.iloc[-2]["HA_Open"] else "BEAR",
            "Trend": trend,
            "RSI": round(last["RSI"], 1),
            "ATR%": round(atr_pct, 2),
            "Compression": compression,
            "ADX": round(adx, 1),
            "Vol_Ratio": round(vol_ratio, 2),
            "Price": round(price, 2),
            "Market_Regime": market_regime
        })

    return pd.DataFrame(results)

# ==========================
# NSE FUTURES DATA
# ==========================

def get_nse_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.nseindia.com"
    })
    try:
        s.get("https://www.nseindia.com", timeout=10)
    except:
        pass
    return s

def download_bhavcopy(path):
    s = get_nse_session()
    for i in range(7):
        d = date.today() - timedelta(days=i)
        ymd = d.strftime("%Y%m%d")
        url = f"https://nsearchives.nseindia.com/content/{path}/BhavCopy_NSE_{path.upper()}_0_0_0_{ymd}_F_0000.csv.zip"
        r = s.get(url)
        if r.status_code == 200:
            with zipfile.ZipFile(BytesIO(r.content)) as z:
                return pd.read_csv(z.open(z.namelist()[0]))
    raise RuntimeError("Bhavcopy not found")

def futures_signal(row):
    oi = row["ChngInOpnIntrst"]
    price = row["ClsPric"] - row["OpnPric"]
    if oi > 0 and price > 0:
        return "Long Buildup"
    if oi > 0 and price < 0:
        return "Short Buildup"
    if oi < 0 and price > 0:
        return "Short Covering"
    return "Long Unwinding"

def get_near_two_futures(fo, symbol):
    inst = "IDF" if symbol in INDEX_SYMBOLS else "STF"
    df = fo[(fo["FinInstrmTp"] == inst) & (fo["TckrSymb"] == symbol)].copy()
    df["XpryDt"] = pd.to_datetime(df["XpryDt"])
    df = df.sort_values("XpryDt")
    return df.head(2)

def run_futures_bias():
    fo = download_bhavcopy("fo")
    rows = []

    for sym in TICKERS:
        futs = get_near_two_futures(fo, sym)
        if len(futs) < 2:
            continue

        f1, f2 = futs.iloc[0], futs.iloc[1]

        rows.append({
            "Ticker": sym,
            "F1_Expiry": f1["XpryDt"].date(),
            "F1_Close": f1["ClsPric"],
            "F1_OI_Change": f1["ChngInOpnIntrst"],
            "F1_OI_%": round((f1["ChngInOpnIntrst"] / max(f1["OpnIntrst"], 1)) * 100, 2),
            "F1_Signal": futures_signal(f1),

            "F2_Expiry": f2["XpryDt"].date(),
            "F2_Close": f2["ClsPric"],
            "F2_OI_Change": f2["ChngInOpnIntrst"],
            "F2_OI_%": round((f2["ChngInOpnIntrst"] / max(f2["OpnIntrst"], 1)) * 100, 2),
            "F2_Signal": futures_signal(f2)
        })

    return pd.DataFrame(rows)

# ==========================
# INTELLIGENCE LAYERS
# ==========================

def detect_futures_flip(f1, f2):
    bullish = ["Long Buildup", "Short Covering"]
    bearish = ["Short Buildup", "Long Unwinding"]
    if f1 in bullish and f2 in bearish:
        return "BEARISH_FLIP"
    if f1 in bearish and f2 in bullish:
        return "BULLISH_FLIP"
    return "NO"

def compute_sl(price, atr_pct, side):
    atr = price * (atr_pct / 100)
    if side == "LONG":
        return round(price - 1.8 * atr, 2), round(price - atr, 2)
    return round(price + 1.8 * atr, 2), round(price + atr, 2)

def strong_buy_requirements(row):
    missing = []
    if row["Compression"] != "YES":
        missing.append("Compression")
    if row["ADX"] < 22:
        missing.append("ADX ≥22")
    if row["Vol_Ratio"] < 1.5:
        missing.append("Volume Expansion")
    if row["Futures_Flip"] != "NO":
        missing.append("Futures Alignment")
    return "ALL CONDITIONS MET" if not missing else " | ".join(missing)

# ==========================
# FINAL SCANNER
# ==========================

def run_scanner():
    cash = run_core_scanner()
    fut = run_futures_bias()

    df = cash.merge(fut, on="Ticker", how="left")

    df["Futures_Flip"] = df.apply(
        lambda r: detect_futures_flip(r["F1_Signal"], r["F2_Signal"]),
        axis=1
    )

    df["Auto_SL"], df["Trailing_SL"] = zip(*df.apply(
        lambda r: compute_sl(r["Price"], r["ATR%"], r["Breakout"]),
        axis=1
    ))

    df["What_Makes_STRONG_BUY"] = df.apply(strong_buy_requirements, axis=1)

    df.loc[df["Futures_Flip"] != "NO", "Final_Score"] -= 1.5

    return df.sort_values("Final_Score", ascending=False)

# ==========================
# RUN
# ==========================

if __name__ == "__main__":
    df = run_scanner()
    print(df.to_string(index=False))
    df.to_csv("scanner_output.csv", index=False)
