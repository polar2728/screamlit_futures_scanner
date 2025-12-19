import pandas as pd
import numpy as np
import requests
import zipfile
import yfinance as yf
from io import BytesIO
from datetime import date, timedelta, datetime
import time as time

# ==========================
# CONFIG
# ==========================

ACCOUNT_CAPITAL = 250000
RISK_PER_TRADE_PCT = 1.0

LOOKBACK_DAYS = "9mo"
RSI_LEN = 14
EMA_FAST = 20
EMA_SLOW = 50
ATR_LEN = 14
DONCHIAN_LEN = 5

INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY"}

# Common stock futures universe
TICKERS = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "AXISBANK", "SBIN",
    "INFY", "TCS", "ITC", "HINDUNILVR", "BHARTIARTL", "LT", "KOTAKBANK",
    "TATASTEEL", "COALINDIA", "ASIANPAINT", "MARUTI", "DRREDDY",
    "TATAPOWER", "INDIGO", "ULTRACEMCO", "ONGC"
]

# Map to Yahoo symbols for scanner
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
    "ONGC": "ONGC.NS"
}

# ==========================
# INDICATORS (scanner)
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
    df["HA_Low"]  = np.minimum(df["Low"],  np.minimum(df["HA_Open"], df["HA_Close"]))
    return df

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()
    rs = avg_gain / avg_loss
    return (100 - (100 / (1 + rs))).fillna(50)

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()

# ==========================
# HA SCANNER (EOD)
# ==========================

def run_ha_scanner() -> pd.DataFrame:
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
            time.sleep(1.5)
            if df.empty or len(df) < 60:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            ha = heikin_ashi(df)
            ha["EMA20"] = ha["HA_Close"].ewm(span=EMA_FAST, adjust=False).mean()
            ha["EMA50"] = ha["HA_Close"].ewm(span=EMA_SLOW, adjust=False).mean()
            ha["RSI"]   = rsi(ha["HA_Close"], RSI_LEN)
            ha["ATR"]   = atr(df, ATR_LEN)

            last = ha.iloc[-1]
            prev = ha.iloc[-2]

            ha_bull = last["HA_Close"] > last["HA_Open"]
            ha_bear = last["HA_Close"] < last["HA_Open"]

            prev_candle = "BULL" if prev["HA_Close"] > prev["HA_Open"] else "BEAR"
            curr_candle = "BULL" if ha_bull else "BEAR"

            daily_up   = last["HA_Close"] > last["EMA20"] > last["EMA50"]
            daily_down = last["HA_Close"] < last["EMA20"] < last["EMA50"]

            vol_ma20 = df["Volume"].rolling(20).mean().iloc[-1]
            vol_ratio = df["Volume"].iloc[-1] / vol_ma20 if vol_ma20 > 0 else 0
            vol_confirm = vol_ratio > 1.2

            body   = abs(last["HA_Close"] - last["HA_Open"])
            range_ = last["HA_High"] - last["HA_Low"]
            is_doji = (body / range_) < 0.2 if range_ > 0 else True

            # Score
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

            don_low  = ha["HA_Low"].rolling(DONCHIAN_LEN).min().shift(1).iloc[-1]
            don_high = ha["HA_High"].rolling(DONCHIAN_LEN).max().shift(1).iloc[-1]

            price = df["Close"].iloc[-1]

            if verdict.endswith("BUY"):
                sl = don_low
            elif verdict.endswith("SELL"):
                sl = don_high
            else:
                sl = np.nan

            dist_to_sl = abs(price - sl) if not np.isnan(sl) else np.nan

            confidence = min(100, abs(score) * 20)

            results.append({
                "Ticker": name,
                "Prev HA": prev_candle,
                "Cur HA": curr_candle,
                "Verdict": verdict,
                "Score": score,
                "Conf%": confidence,
                "Price": round(price, 2),
                "Donchian_SL": round(sl, 2) if not np.isnan(sl) else "—",
                "SL_Distance": round(dist_to_sl, 2) if not np.isnan(dist_to_sl) else "—",
                "ATR": round(last["ATR"], 2),
                "RSI": round(last["RSI"], 2),
                "Trend": "UP" if daily_up else "DOWN" if daily_down else "SIDE",
                "Vol_Ratio": round(vol_ratio, 2)
            })

        except Exception as e:
            print(f"Scanner skipped {name}: {e}")

    if not results:
        return pd.DataFrame()

    report = pd.DataFrame(results)
    return report.sort_values(["Conf%", "Vol_Ratio"], ascending=False)

# ==========================
# NSE UDIFF FUTURES SECTION
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
    except Exception:
        pass
    return session

def try_download_cm_udiff(session, lookback_days=5):
    for i in range(lookback_days):
        d = date.today() - timedelta(days=i)
        yyyymmdd = d.strftime("%Y%m%d")
        url = (
            "https://nsearchives.nseindia.com/content/cm/"
            f"BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip"
        )
        try:
            r = session.get(url, timeout=15)
            if r.status_code == 200 and "zip" in r.headers.get("Content-Type", "").lower():
                with zipfile.ZipFile(BytesIO(r.content)) as z:
                    fname = z.namelist()[0]
                    print(f"Loaded CM bhavcopy (UDiFF) for {d:%Y-%m-%d}")
                    df = pd.read_csv(z.open(fname))
                    return df
        except Exception:
            continue
    raise RuntimeError("No CM UDiFF bhavcopy found in last few days")

def try_download_fo_udiff(session, lookback_days=5):
    for i in range(lookback_days):
        d = date.today() - timedelta(days=i)
        yyyymmdd = d.strftime("%Y%m%d")
        url = (
            "https://nsearchives.nseindia.com/content/fo/"
            f"BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip"
        )
        try:
            r = session.get(url, timeout=15)
            if r.status_code == 200 and "zip" in r.headers.get("Content-Type", "").lower():
                with zipfile.ZipFile(BytesIO(r.content)) as z:
                    fname = z.namelist()[0]
                    print(f"Loaded FO bhavcopy (UDiFF) for {d:%Y-%m-%d}")
                    df = pd.read_csv(z.open(fname))
                    return df
        except Exception:
            continue
    raise RuntimeError("No FO UDiFF bhavcopy found in last few days")

def load_bhavcopies():
    session = get_nse_session()
    fo_df = try_download_fo_udiff(session)
    cm_df = try_download_cm_udiff(session)
    return fo_df, cm_df


def get_near_month_future(fo_df, symbol):
    """
    UDiFF mapping:
    - STF: Futures Stock (FUTSTK)
    - IDF: Index Futures (FUTIDX)
    - TckrSymb: underlying symbol (e.g. RELIANCE, NIFTY)
    - XpryDt: expiry date
    """
    sym = symbol.upper()

    inst_type = "IDF" if sym in INDEX_SYMBOLS else "STF"

    fut = fo_df[
        (fo_df["FinInstrmTp"].astype(str).str.strip().str.upper() == inst_type) &
        (fo_df["TckrSymb"].astype(str).str.strip().str.upper() == sym)
    ].copy()

    if fut.empty:
        return None

    fut["XpryDt"] = pd.to_datetime(fut["XpryDt"])
    fut = fut.sort_values("XpryDt")

    return fut.iloc[0]


def futures_conviction(fut, spot_close):
    fut_close = fut["ClsPric"]
    fut_open  = fut["OpnPric"]
    oi_change = fut["ChngInOpnIntrst"]
    volume    = fut["TtlTradgVol"]

    premium_pct = ((fut_close - spot_close) / spot_close) * 100
    price_change = fut_close - fut_open

    if oi_change > 0 and price_change > 0:
        buildup = "Long Buildup"
        bias = "BULLISH"
    elif oi_change > 0 and price_change < 0:
        buildup = "Short Buildup"
        bias = "BEARISH"
    elif oi_change < 0 and price_change > 0:
        buildup = "Short Covering"
        bias = "BULLISH"
    else:
        buildup = "Long Unwinding"
        bias = "BEARISH"

    return {
        "Fut Close": round(float(fut_close), 2),
        "Spot Close": round(float(spot_close), 2),
        "Premium %": round(float(premium_pct), 2),
        "OI Change": int(oi_change),
        "Volume (Contracts)": int(volume),
        "Buildup": buildup,
        "Bias": bias
    }

def run_futures_conviction():
    fo_df, cm_df = load_bhavcopies()

    cm_symbol_col = "TckrSymb"
    cm_close_col  = "ClsPric"
    if cm_symbol_col not in cm_df.columns or cm_close_col not in cm_df.columns:
        raise RuntimeError(f"Unexpected CM columns: {cm_df.columns.tolist()}")

    results = []
    for symbol in TICKERS:
        try:
            fut = get_near_month_future(fo_df, symbol)
            if fut is None:
                continue
            spot_row = cm_df[cm_df[cm_symbol_col] == symbol]
            if spot_row.empty:
                continue
            spot_close = spot_row.iloc[0][cm_close_col]
            conviction = futures_conviction(fut, spot_close)
            conviction["Symbol"] = symbol
            conviction["Expiry"] = fut["XpryDt"].date()
            results.append(conviction)
        except Exception as e:
            print(f"Skipped {symbol}: {e}")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df[
        ["Symbol", "Expiry", "Spot Close", "Fut Close",
         "Premium %", "OI Change", "Volume (Contracts)", "Buildup", "Bias"]
    ]
    return df

# ==========================
# COMBINED CONVICTION SCORING
# ==========================

def score_scanner_row(verdict: str, score: int) -> int:
    if verdict == "STRONG BUY":
        base = 4
    elif verdict == "WEAK BUY":
        base = 2
    elif verdict == "WEAK SELL":
        base = -2
    elif verdict == "STRONG SELL":
        base = -4
    else:
        base = 0
    return base + max(-2, min(2, score))

def score_futures_row(bias: str, buildup: str) -> float:
    if bias == "BULLISH":
        base = 2
    elif bias == "BEARISH":
        base = -2
    else:
        base = 0

    if buildup == "Long Buildup":
        base += 1
    elif buildup == "Short Covering":
        base += 0.5
    elif buildup == "Short Buildup":
        base -= 1
    elif buildup == "Long Unwinding":
        base -= 0.5

    return base

def label_total_conviction(total: float) -> str:
    if total >= 6:
        return "HIGH BULLISH"
    elif 3 <= total < 6:
        return "MODERATE BULLISH"
    elif 0 < total < 3:
        return "MILD BULLISH"
    elif total == 0:
        return "NEUTRAL"
    elif -3 < total < 0:
        return "MILD BEARISH"
    elif -6 < total <= -3:
        return "MODERATE BEARISH"
    else:
        return "HIGH BEARISH"

# ==========================
# DASHBOARD MERGE
# ==========================

def run_scanner():
    scanner_df = run_ha_scanner()
    fut_df     = run_futures_conviction()

    if scanner_df.empty or fut_df.empty:
        return scanner_df, fut_df, pd.DataFrame()

    merged = scanner_df.merge(
        fut_df,
        left_on="Ticker",
        right_on="Symbol",
        how="inner"
    )

    merged["Scan_Score"] = merged.apply(
        lambda row: score_scanner_row(row["Verdict"], row["Score"]),
        axis=1
    )

        # For rows without futures (e.g. NIFTY/BANKNIFTY), Fut_Score will be 0
    def safe_fut_score(row):
        if pd.isna(row["Bias"]) or pd.isna(row["Buildup"]):
            return 0.0
        return score_futures_row(row["Bias"], row["Buildup"])

    merged["Fut_Score"] = merged.apply(safe_fut_score, axis=1)

    merged["Final_Score"] = merged["Scan_Score"] + merged["Fut_Score"]
    merged["Final_Conviction"] = merged["Final_Score"].apply(label_total_conviction)

    merged = merged.sort_values(
        ["Final_Score", "Conf%", "Vol_Ratio"],
        ascending=[False, False, False]
    )

    merged = merged.sort_values(
        ["Final_Score", "Conf%", "Vol_Ratio"],
        ascending=[False, False, False]
    )

    # Drop columns you don't want in the dashboard
    cols_to_drop = ["Price", "Donchian_SL", "SL_Distance", "ATR", "Symbol"]
    merged = merged.drop(columns=[c for c in cols_to_drop if c in merged.columns])
    return merged