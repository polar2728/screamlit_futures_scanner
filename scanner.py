import pandas as pd
import numpy as np
import requests
import zipfile
import yfinance as yf
from io import BytesIO
from datetime import date, timedelta, datetime
import time as time
import os

# ==========================
# CONFIG - SIMPLIFIED
# ==========================

LOOKBACK_DAYS = "9mo"

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

# ==========================
# CORE 4 SIGNALS SCANNER (with ALL display data)
# ==========================

def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    ha_open = []
    for i in range(len(df)):
        if i == 0:
            ha_open.append((df["Open"].iloc[i] + df["Close"].iloc[i]) / 2)
        else:
            ha_open.append((ha_open[i-1] + df["HA_Close"].iloc[i-1]) / 2)
    df["HA_Open"] = ha_open
    df["HA_High"] = np.maximum(df["High"], np.maximum(df["HA_Open"], df["HA_Close"]))
    df["HA_Low"] = np.minimum(df["Low"], np.minimum(df["HA_Open"], df["HA_Close"]))
    return df

def run_core_scanner() -> pd.DataFrame:
    results = []
    
    for name, symbol in SYMBOL_MAP.items():
        try:
            df = yf.download(symbol, period=LOOKBACK_DAYS, interval="1d", 
                           auto_adjust=True, progress=False)
            time.sleep(0.5)
            if df.empty or len(df) < 50:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            price = df["Close"].iloc[-1]
            
            # === CORE SIGNALS ===
            # 1. DONCHIAN 20 BREAKOUT
            don_high_20 = df["High"].rolling(20).max().shift(1).iloc[-1]
            don_low_20 = df["Low"].rolling(20).min().shift(1).iloc[-1]
            breakout_long = price > don_high_20
            breakout_short = price < don_low_20

            # 2. HEIKIN ASHI
            ha = heikin_ashi(df)
            
            # === ALL INDICATORS FIRST ===
            ha["EMA20"] = ha["HA_Close"].ewm(span=20, adjust=False).mean()
            ha["EMA50"] = ha["HA_Close"].ewm(span=50, adjust=False).mean()
            
            # RSI
            delta = ha["HA_Close"].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta).clip(lower=0).rolling(14).mean()
            rs = gain / loss
            ha["RSI"] = 100 - (100 / (1 + rs)).fillna(50)
            
            # ATR
            tr1 = df["High"] - df["Low"]
            tr2 = (df["High"] - df["Close"].shift()).abs()
            tr3 = (df["Low"] - df["Close"].shift()).abs()
            ha["ATR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
            
            # NOW access last_ha
            last_ha = ha.iloc[-1]
            prev_ha = ha.iloc[-2]
            
            ha_bull = last_ha["HA_Close"] > last_ha["HA_Open"]
            ha_bear = last_ha["HA_Close"] < last_ha["HA_Open"]
            prev_candle = "BULL" if prev_ha["HA_Close"] > prev_ha["HA_Open"] else "BEAR"
            curr_candle = "BULL" if ha_bull else "BEAR"

            # 3. VOLUME
            vol_ma20 = df["Volume"].rolling(20).mean().iloc[-1]
            vol_ratio = df["Volume"].iloc[-1] / vol_ma20 if vol_ma20 > 0 else 0
            vol_confirm = vol_ratio > 1.2

            # 4. DOJI FILTER
            body = abs(last_ha["HA_Close"] - last_ha["HA_Open"])
            range_ = last_ha["HA_High"] - last_ha["HA_Low"]
            is_doji = (body / range_) < 0.2 if range_ > 0 else True

            # === SCORING (UNCHANGED - 4 signals only) ===
            score = 0
            if breakout_long: score += 3
            if breakout_short: score -= 3
            if ha_bull: score += 1
            if ha_bear: score -= 1
            if vol_confirm: score += 0.5
            if is_doji: score -= 0.5

            confidence = min(100, abs(score) * 20)

            # === DISPLAY-ONLY ===
            daily_up = last_ha["HA_Close"] > last_ha["EMA20"] > last_ha["EMA50"]
            daily_down = last_ha["HA_Close"] < last_ha["EMA20"] < last_ha["EMA50"]
            trend = "UP" if daily_up else "DOWN" if daily_down else "SIDE"

            # Donchian SL
            don_low_5 = ha["HA_Low"].rolling(5).min().shift(1).iloc[-1]
            don_high_5 = ha["HA_High"].rolling(5).max().shift(1).iloc[-1]
            sl_long = don_low_5
            sl_short = don_high_5
            dist_to_sl_long = abs(price - sl_long) if not np.isnan(sl_long) else np.nan
            dist_to_sl_short = abs(price - sl_short) if not np.isnan(sl_short) else np.nan

            # === NEW: 3 EXTRA FALSE BREAKOUT FILTERS (DISPLAY ONLY) ===
            # 1. 2-BAR CONFIRMATION
            breakout_confirmed_long = price > don_high_20 and df["Close"].iloc[-2] > don_high_20
            breakout_confirmed_short = price < don_low_20 and df["Close"].iloc[-2] < don_low_20
            breakout_conf = "YES" if (breakout_confirmed_long or breakout_confirmed_short) else "PENDING"

            # 2. NARROW RANGE COMPRESSION (40% contraction before breakout)
            range_20d = don_high_20 - don_low_20 if not np.isnan(don_high_20) and not np.isnan(don_low_20) else np.nan
            recent_range_5d = df["High"].iloc[-5:].max() - df["Low"].iloc[-5:].min()
            compression = recent_range_5d < (range_20d * 0.6) if not np.isnan(range_20d) else False
            compression_status = "YES" if compression else "NO"

            # 3. ADX TREND STRENGTH (pure pandas implementation)
            high_low = df["High"] - df["Low"]
            high_close = (df["High"] - df["Close"].shift()).abs()
            low_close = (df["Low"] - df["Close"].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr_14 = tr.rolling(14).mean()
            
            plus_dm = df["High"].diff()
            minus_dm = df["Low"].diff() * -1
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
            minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
            
            plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / atr_14)
            minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / atr_14)
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(14).mean().iloc[-1]
            
            adx_status = "STRONG" if adx > 25 else "MEDIUM" if adx > 20 else "WEAK"

            results.append({
                "Ticker": name,
                "Score": round(score, 1),
                "Conf%": confidence,
                "Verdict": "STRONG BUY" if score >= 4 else "STRONG SELL" if score <= -4 else "WEAK" if score != 0 else "NEUTRAL",
                "Breakout": "LONG" if breakout_long else "SHORT" if breakout_short else "NONE",
                "HA": curr_candle,
                "Vol_Ratio": round(vol_ratio, 2),
                "Price": round(price, 2),
                
                # ORIGINAL DISPLAY
                "Prev_HA": prev_candle,
                "Trend": trend,
                "RSI": round(last_ha["RSI"], 1) if not np.isnan(last_ha["RSI"]) else "—",
                "ATR": round(last_ha["ATR"], 2) if not np.isnan(last_ha["ATR"]) else "—",
                "DC_SL_Long": round(sl_long, 2) if not np.isnan(sl_long) else "—",
                "DC_SL_Short": round(sl_short, 2) if not np.isnan(sl_short) else "—",
                # === NEW: 3 FALSE BREAKOUT FILTERS ===
                "Breakout_Conf": breakout_conf,
                "Compression": compression_status,
                "ADX_Trend": adx_status
            })

        except Exception as e:
            print(f"Scanner skipped {name}: {e}")

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values(["Score", "Conf%"], ascending=False)


# ==========================
# FUTURES (unchanged from streamlined version)
# ==========================

def get_nse_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com"
    })
    try:
        session.get("https://www.nseindia.com", timeout=10)
    except:
        pass
    return session

def try_download_fo_udiff(session, lookback_days=5):
    for i in range(lookback_days):
        d = date.today() - timedelta(days=i)
        yyyymmdd = d.strftime("%Y%m%d")
        url = f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip"
        try:
            r = session.get(url, timeout=15)
            if r.status_code == 200 and "zip" in r.headers.get("Content-Type", "").lower():
                with zipfile.ZipFile(BytesIO(r.content)) as z:
                    fname = z.namelist()[0]
                    print(f"Loaded FO bhavcopy for {d:%Y-%m-%d}")
                    return pd.read_csv(z.open(fname))
        except:
            continue
    raise RuntimeError("No FO bhavcopy found")

def try_download_cm_udiff(session, lookback_days=5):
    for i in range(lookback_days):
        d = date.today() - timedelta(days=i)
        yyyymmdd = d.strftime("%Y%m%d")
        url = f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip"
        try:
            r = session.get(url, timeout=15)
            if r.status_code == 200 and "zip" in r.headers.get("Content-Type", "").lower():
                with zipfile.ZipFile(BytesIO(r.content)) as z:
                    fname = z.namelist()[0]
                    print(f"Loaded CM bhavcopy for {d:%Y-%m-%d}")
                    return pd.read_csv(z.open(fname))
        except:
            continue
    raise RuntimeError("No CM bhavcopy found")

def get_near_and_next_future(fo_df, symbol):
    sym = symbol.upper()
    inst_type = "IDF" if sym in INDEX_SYMBOLS else "STF"
    
    fut = fo_df[
        (fo_df["FinInstrmTp"].astype(str).str.strip().str.upper() == inst_type) &
        (fo_df["TckrSymb"].astype(str).str.strip().str.upper() == sym)
    ].copy()
    
    if fut.empty:
        return None, None
    
    fut["XpryDt"] = pd.to_datetime(fut["XpryDt"])
    fut = fut.sort_values("XpryDt")
    near = fut.iloc[0]
    nextm = fut.iloc[1] if len(fut) > 1 else None
    return near, nextm

def futures_conviction(fut, spot_close):
    oi_change = fut["ChngInOpnIntrst"]
    price_change = fut["ClsPric"] - fut["OpnPric"]
    
    if oi_change > 0 and price_change > 0:
        bias_score, bias = 2.0, "BULLISH"
    elif oi_change > 0 and price_change < 0:
        bias_score, bias = -2.0, "BEARISH"
    elif oi_change < 0 and price_change > 0:
        bias_score, bias = 1.0, "BULLISH"
    else:
        bias_score, bias = -1.0, "BEARISH"
    
    return {
        "Fut_Score": bias_score,
        "Fut_Bias": bias,
        "OI_Change": int(oi_change),
        "Fut_Close": round(float(fut["ClsPric"]), 2),
        "Expiry": fut["XpryDt"].date()
    }

def run_futures_bias():
    session = get_nse_session()
    fo_df = try_download_fo_udiff(session)
    cm_df = try_download_cm_udiff(session)
    
    results = []
    for symbol in TICKERS:
        try:
            near, nextm = get_near_and_next_future(fo_df, symbol)
            if near is None:
                continue
                
            spot_row = cm_df[cm_df["TckrSymb"] == symbol]
            if spot_row.empty:
                continue
            spot_close = spot_row.iloc[0]["ClsPric"]
            
            near_data = futures_conviction(near, spot_close)
            near_data["Symbol"] = symbol
            
            # Next month (display only)
            if nextm is not None:
                next_data = futures_conviction(nextm, spot_close)
                near_data.update({
                    "Next_Expiry": next_data["Expiry"],
                    "Next_Fut_Close": next_data["Fut_Close"],
                    "Next_Fut_Bias": next_data["Fut_Bias"],
                    "Next_OI_Change": next_data["OI_Change"]
                })
            
            results.append(near_data)
        except:
            continue
    
    return pd.DataFrame(results)

# ==========================
# FINAL DASHBOARD (8 CORE + 12 DISPLAY)
# ==========================

def run_scanner():
    print("=== CORE SCANNER ===")
    scanner_df = run_core_scanner()
    print("=== FUTURES BIAS ===")
    fut_df = run_futures_bias()
    
    if scanner_df.empty:
        return pd.DataFrame()
    
    # Merge futures
    if not fut_df.empty:
        merged = scanner_df.merge(fut_df, left_on="Ticker", right_on="Symbol", how="left")
        merged["Final_Score"] = merged["Score"] + merged["Fut_Score"].fillna(0)
        merged["Fut_Bias"] = merged["Fut_Bias"].fillna("NO FUT")
    else:
        merged = scanner_df.copy()
        merged["Final_Score"] = merged["Score"]
        merged["Fut_Bias"] = "NO DATA"
    
    # Final verdict
    def get_final_verdict(row):
        score = row["Final_Score"]
        if score >= 5.0:
            return "STRONG BUY"
        elif score <= -5.0:
            return "STRONG SELL"
        elif score > 1.0:
            return "WEAK BUY"
        elif score < -1.0:
            return "WEAK SELL"
        else:
            return "NEUTRAL"
    
    merged["Final_Verdict"] = merged.apply(get_final_verdict, axis=1)
    merged["Conf%"] = (abs(merged["Final_Score"]) * 15).clip(upper=100).round(0)
    
    # === 8 PINNED CORE COLUMNS + DISPLAY COLUMNS ===
    core_cols = [
        "Ticker", "Final_Score", "Final_Verdict", "Breakout", "HA", 
        "Fut_Bias", "Vol_Ratio", "Conf%"
    ]
    
    display_cols = [
        "Prev_HA", "Trend", "RSI", "ATR",
        "Price", "DC_SL_Long", "DC_SL_Short", "Breakout_Conf", "Compression",
        "ADX_Trend", "Expiry", "Fut_Close", "OI_Change",
        "Next_Expiry", "Next_Fut_Close", "Next_Fut_Bias", "Next_OI_Change"
    ]
    
    # All columns in order
    all_cols = [c for c in core_cols if c in merged.columns] + \
               [c for c in display_cols if c in merged.columns]
    
    return merged[all_cols].sort_values("Final_Score", ascending=False)

if __name__ == "__main__":
    dashboard = run_scanner()
    if not dashboard.empty:
        print("\n=== 4-SIGNAL DASHBOARD (8 CORE + 12 DISPLAY) ===")
        print(dashboard.to_string(index=False))
        dashboard.to_csv("complete_dashboard.csv", index=False)
        strong_buys = len(dashboard[dashboard['Final_Score'] >= 5])
        strong_sells = len(dashboard[dashboard['Final_Score'] <= -5])
        print(f"\nSTRONG BUYS: {strong_buys} | STRONG SELLS: {strong_sells}")
    else:
        print("No signals today.")
