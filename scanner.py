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

TICKERS = [
    "NIFTY", "BANKNIFTY", "RELIANCE", "HDFCBANK", "ICICIBANK", "AXISBANK", "SBIN",
    "INFY", "TCS", "ITC", "HINDUNILVR", "BHARTIARTL", "LT", "KOTAKBANK",
    "TATASTEEL", "COALINDIA", "ASIANPAINT", "MARUTI", "DRREDDY",
    "TATAPOWER", "INDIGO", "ULTRACEMCO", "ONGC", "BAJFINANCE"
]

SYMBOL_MAP = {t: (f"{t}.NS" if t not in {"NIFTY","BANKNIFTY"} else "^NSEI" if t=="NIFTY" else "^NSEBANK") for t in TICKERS}

# ==========================================================
# SAFE YFINANCE
# ==========================================================
def safe_yf_download(symbol):
    for _ in range(MAX_RETRIES):
        try:
            df = yf.download(symbol, period=LOOKBACK_DAYS, interval="1d",
                             auto_adjust=True, progress=False, threads=False)
            if not df.empty and len(df) > 50:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df.reset_index()
        except:
            time.sleep(1)
    return pd.DataFrame()

# ==========================================================
# HEIKIN ASHI
# ==========================================================
def heikin_ashi(df):
    ha = df.copy()
    ha["HA_Close"] = ha[["Open","High","Low","Close"]].mean(axis=1)
    ha_open = [ (ha["Open"].iloc[0] + ha["Close"].iloc[0]) / 2 ]
    for i in range(1,len(ha)):
        ha_open.append((ha_open[i-1] + ha["HA_Close"].iloc[i-1]) / 2)
    ha["HA_Open"] = ha_open
    ha["HA_High"] = ha[["High","HA_Open","HA_Close"]].max(axis=1)
    ha["HA_Low"]  = ha[["Low","HA_Open","HA_Close"]].min(axis=1)
    return ha

# ==========================================================
# CASH SCANNER
# ==========================================================
def run_core_scanner():
    results = []

    nifty = safe_yf_download("^NSEI")
    market_regime = "NEUTRAL"
    if not nifty.empty:
        ema20 = nifty["Close"].ewm(20).mean().iloc[-1]
        ema50 = nifty["Close"].ewm(50).mean().iloc[-1]
        market_regime = "RISK_ON" if ema20 > ema50 else "RISK_OFF"

    for name, symbol in SYMBOL_MAP.items():
        df = safe_yf_download(symbol)
        if df.empty:
            continue

        price = df["Close"].iloc[-1]

        # Donchian breakout
        dh = df["High"].rolling(20).max().shift(1).iloc[-1]
        dl = df["Low"].rolling(20).min().shift(1).iloc[-1]
        breakout = "LONG" if price > dh else "SHORT" if price < dl else "NONE"

        ha = heikin_ashi(df)
        ha["EMA20"] = ha["HA_Close"].ewm(20).mean()
        ha["EMA50"] = ha["HA_Close"].ewm(50).mean()

        trend = "UP" if ha["EMA20"].iloc[-1] > ha["EMA50"].iloc[-1]*1.02 else \
                "DOWN" if ha["EMA20"].iloc[-1] < ha["EMA50"].iloc[-1]*0.98 else "SIDE"

        ha_today = "BULL" if ha["HA_Close"].iloc[-1] > ha["HA_Open"].iloc[-1] else "BEAR"
        prev_ha = "BULL" if ha["HA_Close"].iloc[-2] > ha["HA_Open"].iloc[-2] else "BEAR"

        # RSI
        delta = ha["HA_Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta).clip(lower=0).rolling(14).mean()
        rsi = 100 - (100/(1+(gain/(loss+1e-9)))).iloc[-1]

        # ATR
        tr = pd.concat([
            df["High"]-df["Low"],
            (df["High"]-df["Close"].shift()).abs(),
            (df["Low"]-df["Close"].shift()).abs()
        ],axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        atr_pct = atr/price*100

        # ADX
        adx = talib.ADX(df["High"],df["Low"],df["Close"],14).iloc[-1]
        compression = "YES" if atr_pct < 0.8 and adx < 20 else "NO"

        vol_ratio = df["Volume"].iloc[-1] / df["Volume"].rolling(20).mean().iloc[-1]

        score = 0
        score += 3 if breakout=="LONG" else -3 if breakout=="SHORT" else 0
        score += 1 if ha_today=="BULL" else -1
        if atr_pct < MIN_ATR_PCT:
            score *= 0.5

        results.append({
            "Ticker": name,
            "Final_Score": round(score,2),
            "Final_Verdict": "STRONG BUY" if score>=5 else
                             "STRONG SELL" if score<=-5 else
                             "WEAK BUY" if score>1 else
                             "WEAK SELL" if score<-1 else "NEUTRAL",
            "Breakout": breakout,
            "HA": ha_today,
            "Prev_HA": prev_ha,
            "Trend": trend,
            "RSI": round(rsi,1),
            "ATR%": round(atr_pct,2),
            "Compression": compression,
            "ADX": round(adx,1),
            "Vol_Ratio": round(vol_ratio,2),
            "Price": round(price,2),
            "Market_Regime": market_regime
        })

    return pd.DataFrame(results)

# ==========================================================
# FUTURES BHAVCOPY
# ==========================================================
def get_nse_session():
    s = requests.Session()
    s.headers.update({"User-Agent":"Mozilla/5.0","Referer":"https://www.nseindia.com"})
    s.get("https://www.nseindia.com", timeout=5)
    return s

def download_fo():
    s = get_nse_session()
    for i in range(7):
        d = date.today()-timedelta(i)
        url = f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv.zip"
        r = s.get(url)
        if r.status_code==200:
            with zipfile.ZipFile(BytesIO(r.content)) as z:
                return pd.read_csv(z.open(z.namelist()[0]))
    return pd.DataFrame()

def futures_signal(row):
    pc = row["ClsPric"]-row["OpnPric"]
    oi = row["ChngInOpnIntrst"]
    if pc>0 and oi>0: return "Long Buildup"
    if pc<0 and oi>0: return "Short Buildup"
    if pc>0 and oi<0: return "Short Covering"
    return "Long Unwinding"

def run_futures():
    fo = download_fo()
    if fo.empty:
        return pd.DataFrame()

    out=[]
    for sym in TICKERS:
        f = fo[(fo["TckrSymb"]==sym)&(fo["FinInstrmTp"].isin(["STF","IDF"]))].copy()
        if f.empty: continue
        f["XpryDt"]=pd.to_datetime(f["XpryDt"])
        f=f.sort_values("XpryDt").head(2)

        row={"Ticker":sym}
        for i,(_,r) in enumerate(f.iterrows(),1):
            row.update({
                f"F{i}_Expiry": r["XpryDt"].date(),
                f"F{i}_Close": r["ClsPric"],
                f"F{i}_OI_Change": r["ChngInOpnIntrst"],
                f"F{i}_OI_%": round((r["ChngInOpnIntrst"]/r["OpnIntrst"])*100,2) if r["OpnIntrst"]>0 else 0,
                f"F{i}_Signal": futures_signal(r)
            })
        out.append(row)

    return pd.DataFrame(out)

# ==========================================================
# FINAL
# ==========================================================
def run_scanner():
    cash = run_core_scanner()
    fut = run_futures()
    df = cash.merge(fut,on="Ticker",how="left")
    return df.sort_values("Final_Score",ascending=False)

if __name__=="__main__":
    df = run_scanner()
    print(df.to_string(index=False))
    df.to_csv("scanner_output.csv",index=False)
