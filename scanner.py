# scanner.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ==========================
# CONFIG
# ==========================
LOOKBACK_DAYS = "9mo"
RSI_LEN = 14
EMA_FAST = 20
EMA_SLOW = 50
ATR_LEN = 14
DONCHIAN_LEN = 5

ACCOUNT_CAPITAL = 250000
RISK_PER_TRADE_PCT = 1.0  # 1%

SYMBOL_MAP = {
    "LT": "LT.NS",
    "INDIGO": "INDIGO.NS",
    "AXISBANK": "AXISBANK.NS",
    "GOLD": "GC=F",
    "RELIANCE": "RELIANCE.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "ULTRACEMCO": "ULTRACEMCO.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "MARUTI": "MARUTI.NS",
    "NIFTY": "^NSEI",
    "BHARTIARTL": "BHARTIARTL.NS",
    "TATAPOWER": "TATAPOWER.NS",
    "DRREDDY": "DRREDDY.NS",
    "ONGC": "ONGC.NS",
    "BANKNIFTY": "^NSEBANK",
    "HDFCBANK": "HDFCBANK.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "SBI": "SBIN.NS"
}

# ==========================
# INDICATORS
# ==========================
def heikin_ashi(df):
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

def rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()
    rs = avg_gain / avg_loss
    return (100 - (100 / (1 + rs))).fillna(50)

def atr(df, length=14):
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()

# ==========================
# SCANNER
# ==========================
results = []
scan_time = datetime.now()

for name, symbol in SYMBOL_MAP.items():
    try:
        df = yf.download(
            symbol,
            period=LOOKBACK_DAYS,
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if df.empty or len(df) < 60:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        ha = heikin_ashi(df)
        ha["EMA20"] = ha["HA_Close"].ewm(span=EMA_FAST, adjust=False).mean()
        ha["EMA50"] = ha["HA_Close"].ewm(span=EMA_SLOW, adjust=False).mean()
        ha["RSI"] = rsi(ha["HA_Close"], RSI_LEN)
        ha["ATR"] = atr(df, ATR_LEN)

        last = ha.iloc[-1]
        prev = ha.iloc[-2]

        ha_bull = last["HA_Close"] > last["HA_Open"]
        ha_bear = last["HA_Close"] < last["HA_Open"]

        prev_candle = "BULL" if prev["HA_Close"] > prev["HA_Open"] else "BEAR"
        curr_candle = "BULL" if ha_bull else "BEAR"

        daily_up = last["HA_Close"] > last["EMA20"] > last["EMA50"]
        daily_down = last["HA_Close"] < last["EMA20"] < last["EMA50"]

        vol_ma20 = df["Volume"].rolling(20).mean().iloc[-1]
        vol_ratio = df["Volume"].iloc[-1] / vol_ma20 if vol_ma20 > 0 else 0
        vol_confirm = vol_ratio > 1.2

        body = abs(last["HA_Close"] - last["HA_Open"])
        range_ = last["HA_High"] - last["HA_Low"]
        is_doji = (body / range_) < 0.2 if range_ > 0 else True

        # ===== SCORE =====
        score = 0
        if daily_up: score += 2
        if daily_down: score -= 2
        if ha_bull: score += 1
        if ha_bear: score -= 1
        if last["RSI"] > 60: score += 1
        if last["RSI"] < 40: score -= 1
        if vol_confirm: score += 1
        if is_doji: score -= 1

        if score >= 4:
            verdict = "STRONG BUY"
        elif score <= -4:
            verdict = "STRONG SELL"
        elif score > 0:
            verdict = "WEAK BUY"
        elif score < 0:
            verdict = "WEAK SELL"
        else:
            verdict = "NO TRADE"

        if is_doji:
            verdict = "NO TRADE"

        don_low = ha["HA_Low"].rolling(DONCHIAN_LEN).min().shift(1).iloc[-1]
        don_high = ha["HA_High"].rolling(DONCHIAN_LEN).max().shift(1).iloc[-1]

        price = df["Close"].iloc[-1]

        if verdict.endswith("BUY"):
            sl = don_low
        elif verdict.endswith("SELL"):
            sl = don_high
        else:
            sl = np.nan

        dist_to_sl = abs(price - sl) if not np.isnan(sl) else np.nan
        risk_amt = ACCOUNT_CAPITAL * (RISK_PER_TRADE_PCT / 100)
        pos_units = round(risk_amt / dist_to_sl, 2) if dist_to_sl and dist_to_sl > 0 else 0

        confidence = min(100, abs(score) * 20)

        results.append({
            "Scan_Time": scan_time,
            "Instrument": name,
            "Prev_Candle": prev_candle,
            "Current_Candle": curr_candle,
            "Verdict": verdict,
            "Score": score,
            "Confidence_%": confidence,
            "Price": round(price, 2),
            "Donchian_SL": round(sl, 2) if not np.isnan(sl) else "—",
            "SL_Distance": round(dist_to_sl, 2) if not np.isnan(dist_to_sl) else "—",
            "ATR": round(last["ATR"], 2),
            "Position_Units": pos_units,
            "RSI": round(last["RSI"], 2),
            "Trend": "UP" if daily_up else "DOWN" if daily_down else "SIDE",
            "Vol_Ratio": round(vol_ratio, 2)
        })

    except Exception as e:
        print(f"Skipped {name}: {e}")

# ==========================
# REPORT
# ==========================
report = pd.DataFrame(results)

if report.empty:
    print("No valid signals today.")
else:
    report = report.sort_values(["Confidence_%", "Vol_Ratio"], ascending=False)
    print(report.to_string(index=False))

    fname = f"HA_Daily_Scanner_{scan_time.strftime('%Y%m%d_%H%M')}.csv"
    report.to_csv(fname, index=False)
    print(f"\nSaved: {fname}")
