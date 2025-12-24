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
# GLOBAL CONFIG
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
SYMBOL_MAP = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "RELIANCE": "RELIANCE.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "AXISBANK": "AXISBANK.NS",
    "SBIN": "SBIN.NS",
    "INFY": "INFY.NS",
    "TCS": "TCS.NS",
    "ITC": "ITC.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "LT": "LT.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "TATASTEEL": "TATASTEEL.NS",
    "COALINDIA": "COALINDIA.NS",
    "ASIANPAINT": "ASIANPAINT.NS",
    "MARUTI": "MARUTI.NS",
    "DRREDDY": "DRREDDY.NS",
    "TATAPOWER": "TATAPOWER.NS",
    "INDIGO": "INDIGO.NS",
    "ULTRACEMCO": "ULTRACEMCO.NS",
    "ONGC": "ONGC.NS",
    "BAJFINANCE": "BAJFINANCE.NS"
}


# ==========================================================
# SAFE YFINANCE
# ==========================================================
def safe_yf_download(symbol):
    for attempt in range(MAX_RETRIES):
        try:
            df = yf.download(
                symbol,
                period=LOOKBACK_DAYS,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False
            )
            if not df.empty and len(df) >= 50:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df.reset_index()
        except Exception as e:
            print(f"Retry {attempt+1} failed for {symbol}: {e}")
            time.sleep(1)
    print(f"Failed to download {symbol} after {MAX_RETRIES} retries")
    return pd.DataFrame()


# ==========================================================
# HEIKIN ASHI
# ==========================================================
def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["HA_Close"] = df[["Open", "High", "Low", "Close"]].mean(axis=1)
    
    ha_open = np.empty(len(df))
    ha_open[0] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i-1] + df["HA_Close"].iloc[i-1]) / 2
    
    df["HA_Open"] = ha_open
    df["HA_High"] = df[["High", "HA_Open", "HA_Close"]].max(axis=1)
    df["HA_Low"] = df[["Low", "HA_Open", "HA_Close"]].min(axis=1)
    return df


# ==========================================================
# ENHANCED CASH SCANNER (WITH HA_COLORS)
# ==========================================================
def run_core_scanner() -> pd.DataFrame:
    results = []
    
    # Market Regime from Nifty
    nifty_df = safe_yf_download("^NSEI")
    market_regime = "NEUTRAL"
    if not nifty_df.empty:
        close = nifty_df["Close"]
        ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        market_regime = "RISK_ON" if ema20 > ema50 else "RISK_OFF"
    
    for name, symbol in SYMBOL_MAP.items():
        df = safe_yf_download(symbol)
        if df.empty:
            continue
        
        price = df["Close"].iloc[-1]
        
        # Donchian Channel
        don_high = df["High"].rolling(20).max().shift(1).iloc[-1]
        don_low = df["Low"].rolling(20).min().shift(1).iloc[-1]
        breakout_long = price > don_high if not np.isnan(don_high) else False
        breakout_short = price < don_low if not np.isnan(don_low) else False
        
        # HA Analysis + COLORS (NEW)
        ha = heikin_ashi(df)
        last3 = ha[["HA_Close", "HA_Open"]].tail(3)
        ha_colors = last3["HA_Close"] > last3["HA_Open"]
        ha_color_prev2, ha_color_prev1, ha_color_today = ha_colors
        
        # HA_COLORS column: "RRG" = Red-Red-Green (bull reversal), etc.
        ha_color_str = f"{0 if not ha_color_prev2 else 1}{0 if not ha_color_prev1 else 1}{1 if ha_color_today else 0}"
        ha_pattern = {
            "001": "RRG", "110": "GGR", "111": "GGG", "000": "RRR",
            "010": "RGR", "100": "GRR", "011": "GRG", "101": "RGG"
        }.get(ha_color_str, "MIXED")
        
        bull_reversal = (not ha_color_prev2) and (not ha_color_prev1) and ha_color_today
        bear_reversal = ha_color_prev2 and ha_color_prev1 and (not ha_color_today)
        
        ha_today = "BULL" if ha_color_today else "BEAR"
        prev_ha = "BULL" if ha_color_prev1 else "BEAR"
        
        # Indicators
        delta = ha["HA_Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta).clip(lower=0).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        atr_pct = (atr / price) * 100
        
        vol_ratio = df["Volume"].iloc[-1] / df["Volume"].rolling(20).mean().iloc[-1] if not df["Volume"].rolling(20).mean().iloc[-1] == 0 else 1
        
        # ADX
        adx = talib.ADX(df["High"].values, df["Low"].values, df["Close"].values, timeperiod=14)[-1]
        adx_trend = "STRONG" if adx > 25 else "WEAK"
        
        # Trend
        ha["EMA20"] = ha["HA_Close"].ewm(span=20, adjust=False).mean()
        ha["EMA50"] = ha["HA_Close"].ewm(span=50, adjust=False).mean()
        ha_ema20 = ha["EMA20"].iloc[-1]
        ha_ema50 = ha["EMA50"].iloc[-1]
        trend = "UP" if ha_ema20 > ha_ema50 * 1.02 else "DOWN" if ha_ema20 < ha_ema50 * 0.98 else "SIDE"
        
        compression = "YES" if atr_pct < 0.8 and adx < 20 else "NO"
        breakout_conf = 80 if breakout_long else 20 if breakout_short else 30
        
        # Score
        score = 0
        if breakout_long:
            score += 3
            breakout = "LONG"
        elif breakout_short:
            score -= 3
            breakout = "SHORT"
        else:
            breakout = "NONE"
        
        if bull_reversal:
            score += 2
        elif bear_reversal:
            score -= 2
        elif ha_color_today:
            score += 1
        else:
            score -= 1
        
        if atr_pct < MIN_ATR_PCT:
            score *= 0.5
        
        conf_pct = max(0, min(100, abs(score) * 15))
        
        results.append({
            "Ticker": name,
            "Score": round(score, 2),
            "Price": round(price, 2),
            "ATR": round(atr_pct, 2),
            "RSI": round(rsi, 1),
            "HA": ha_today,
            "Prev_HA": prev_ha,
            "HA_Colors": ha_pattern,  # NEW: Visual HA pattern
            "Trend": trend,
            "Breakout": breakout,
            "Breakout_Conf": breakout_conf,
            "Compression": compression,
            "ADX_Trend": adx_trend,
            "Vol_Ratio": round(vol_ratio, 2),
            "Conf%": round(conf_pct, 1)
        })
    
    return pd.DataFrame(results)


# ==========================================================
# ENHANCED FUTURES WITH OI TRENDS
# ==========================================================
def get_nse_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.nseindia.com"
    })
    try:
        s.get("https://www.nseindia.com", timeout=10)
    except:
        pass
    return s


def download_fo_udiff(session):
    for i in range(7):
        d = date.today() - timedelta(days=i)
        yyyymmdd = d.strftime("%Y%m%d").upper()
        url = f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip"
        try:
            r = session.get(url, timeout=15)
            if r.status_code == 200:
                with zipfile.ZipFile(BytesIO(r.content)) as z:
                    fname = z.namelist()[0]
                    print(f"Loaded FO UDiFF for {d}")
                    return pd.read_csv(z.open(fname))
        except Exception as e:
            print(f"FO attempt {i}: {e}")
            continue
    return pd.DataFrame()


def get_futures_by_expiry(fo_df, symbol):
    inst_type = "IDF" if symbol in INDEX_SYMBOLS else "STF"
    mask = (
        (fo_df["FinInstrmTp"].str.upper() == inst_type) &
        (fo_df["TckrSymb"].str.upper() == symbol)
    )
    df = fo_df[mask].copy()
    if df.empty:
        return []
    df["XpryDt"] = pd.to_datetime(df["XpryDt"])
    return df.sort_values("XpryDt").to_dict('records')


def get_oi_trend(fut_row):
    """Visual OI trend classification"""
    price_chg = fut_row["ClsPric"] - fut_row["OpnPric"]
    oi_chg = fut_row["ChngInOpnIntrst"]
    oi_pct = abs(oi_chg / fut_row["OpnIntrst"] * 100) if fut_row["OpnIntrst"] > 0 else 0
    
    if oi_chg > 0.02 * fut_row["OpnIntrst"] and price_chg > 0:
        return "LONG BUILDUP", 2
    elif oi_chg > 0.02 * fut_row["OpnIntrst"] and price_chg < 0:
        return "SHORT BUILDUP", -2
    elif oi_chg < -0.02 * fut_row["OpnIntrst"] and price_chg > 0:
        return "LONG UNWIND", 1
    elif oi_chg < -0.02 * fut_row["OpnIntrst"] and price_chg < 0:
        return "SHORT COVERING", -1
    elif oi_chg > 0:
        return "LONG ACCUM", 1
    elif oi_chg < 0:
        return "SHORT ACCUM", -1
    else:
        return "NEUTRAL", 0


def run_futures_bias():
    session = get_nse_session()
    fo_df = download_fo_udiff(session)
    if fo_df.empty:
        return pd.DataFrame()
    
    results = []
    for sym in TICKERS:
        futures = get_futures_by_expiry(fo_df, sym)
        if len(futures) >= 1:
            near_fut = futures[0]
            oi_trend, score = get_oi_trend(near_fut)
            
            fut_data = {
                "Ticker": sym,
                "Fut_Score": score,
                "Fut_Bias": oi_trend,  # Visual OI trend
                "Expiry": near_fut["XpryDt"].date(),
                "Fut_Close": round(near_fut["ClsPric"], 2),
                "OI_Change": near_fut["ChngInOpnIntrst"]
            }
            
            if len(futures) > 1:
                next_fut = futures[1]
                next_trend, _ = get_oi_trend(next_fut)
                fut_data.update({
                    "Next_Expiry": next_fut["XpryDt"].date(),
                    "Next_Fut_Close": round(next_fut["ClsPric"], 2),
                    "Next_OI_Change": next_fut["ChngInOpnIntrst"],
                    "Next_Fut_Bias": next_trend
                })
            
            results.append(fut_data)
    
    return pd.DataFrame(results)


# ==========================================================
# FINAL DASHBOARD
# ==========================================================
def run_scanner():
    print("Running enhanced cash scanner...")
    cash_df = run_core_scanner()
    print("Running OI trend futures...")
    try:
        fut_df = run_futures_bias()
    except Exception as e:
        print(f"Futures failed: {e}")
        fut_df = pd.DataFrame()
    
    if cash_df.empty:
        return pd.DataFrame()
    
    df = cash_df.merge(fut_df, on="Ticker", how="left")
    df["Fut_Score"] = df["Fut_Score"].fillna(0)
    df["Fut_Bias"] = df["Fut_Bias"].fillna("NO FUT")
    df["Final_Score"] = df["Score"] + df["Fut_Score"]
    
    conditions = [
        df["Final_Score"] >= 5,
        df["Final_Score"] <= -5,
        df["Final_Score"] > 1,
        df["Final_Score"] < -1
    ]
    choices = ["STRONG BUY", "STRONG SELL", "WEAK BUY", "WEAK SELL"]
    df["Final_Verdict"] = np.select(conditions, choices, default="NEUTRAL")
    
    column_order = [
        "Ticker", "Final_Score", "Final_Verdict", "Breakout", "HA", "Fut_Bias", 
        "Vol_Ratio", "Conf%", "Prev_HA", "HA_Colors", "Trend", "RSI", "ATR", "Price", 
        "Breakout_Conf", "Compression", "ADX_Trend", 
        "Expiry", "Fut_Close", "OI_Change", "Next_Expiry", "Next_Fut_Close", 
        "Next_Fut_Bias", "Next_OI_Change"
    ]
    
    available_cols = [col for col in column_order if col in df.columns]
    df = df[available_cols]
    
    return df.sort_values("Final_Score", ascending=False)


if __name__ == "__main__":
    dashboard = run_scanner()
    if not dashboard.empty:
        print("\n=== ENHANCED SCANNER WITH HA_COLORS & OI TRENDS ===")
        print(dashboard.to_string(index=False))
        dashboard.to_csv("scanner_output.csv", index=False)
        print(f"\nSaved to scanner_output.csv | Rows: {len(dashboard)}")
    else:
        print("No data returned.")
