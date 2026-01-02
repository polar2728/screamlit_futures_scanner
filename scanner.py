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
USE_ALL_FNO = False
BONUS_VOL_RATIO = True

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
            time.sleep(0.4)
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
# ANALYZE CASH (vol_ratio moved up to fix UnboundLocalError)
# ==========================================================
def analyze_cash(name, symbol, market_regime, nifty_return):
    df = safe_yf_download(symbol)
    if df.empty or len(df) < 50:
        return None

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

    # === DOJI & REVERSAL DETECTION ===
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1].item()
    doji_threshold = atr * 0.21

    ha_body_today = abs(ha_close_today - ha_open_today)
    ha_body_prev = abs(ha_close_prev - ha_open_prev)

    is_doji_today = ha_body_today <= doji_threshold
    is_doji_prev = ha_body_prev <= doji_threshold

    # Labels
    ha_today = "DOJI" if is_doji_today else ("BULL" if ha_close_today > ha_open_today else "BEAR")
    prev_ha = "DOJI" if is_doji_prev else ("BULL" if ha_close_prev > ha_open_prev else "BEAR")

    # Base HA contribution
    ha_contribution = 0
    if not is_doji_today:
        ha_contribution = 1 if ha_today == "BULL" else -1

    # Large body bonus
    score_add_body = 0
    if ha_body_today > atr * 0.5 and not is_doji_today:
        score_add_body = 0.5 if ha_today == "BULL" else -0.5

    # Consecutive Doji penalty
    consecutive_doji_penalty = -1.0 if is_doji_today and is_doji_prev else 0

    # === RSI, ATR%, ADX, Compression ===
    delta = ha["HA_Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    rsi = 100 - (100 / (1 + (gain / (loss + 1e-9)))).iloc[-1].item()

    atr_pct = (atr / price) * 100
    adx = talib.ADX(df["High"].values, df["Low"].values, df["Close"].values, timeperiod=14)[-1]
    compression = "YES" if atr_pct < 0.8 and adx < 20 else "NO"

    # === VOL_RATIO (moved up — now available for reversal_vol_bonus) ===
    vol_mean = df["Volume"].rolling(20).mean().iloc[-1].item()
    vol_ratio = df["Volume"].iloc[-1].item() / vol_mean if vol_mean > 0 else 0

    # === REVERSAL CANDLE BONUS ===
    reversal_bonus = 0
    is_reversal_day = False
    if not is_doji_today and ha_body_today > atr * 0.5:
        prev_is_opposite = (prev_ha == "BEAR" and ha_today == "BULL") or (prev_ha == "BULL" and ha_today == "BEAR")
        if is_doji_prev or prev_is_opposite:
            reversal_bonus = 1.5
            is_reversal_day = True

    reversal_vol_bonus = 1.0 if is_reversal_day and vol_ratio > 2 else 0

    # === RELATIVE STRENGTH, RANGE POSITION, STRENGTH TIER ===
    rs_vs_nifty = "N/A"
    if len(df) > 60 and nifty_return is not None:
        try:
            stock_return = (price / df["Close"].iloc[-60]) - 1
            if stock_return > nifty_return * 1.2:
                rs_vs_nifty = "STRONG"
            elif stock_return > nifty_return:
                rs_vs_nifty = "GOOD"
            else:
                rs_vs_nifty = "WEAK"
        except:
            rs_vs_nifty = "N/A"

    range_position = "N/A"
    if len(df) >= 60:
        range_high = df["High"].rolling(60).max().iloc[-1]
        range_low = df["Low"].rolling(60).min().iloc[-1]
        if range_high > range_low:
            position = (price - range_low) / (range_high - range_low)
            range_position = "UPPER" if position > 0.75 else "MIDDLE" if position > 0.4 else "LOWER"

    strength_points = 0
    if breakout == "LONG": strength_points += 3
    elif breakout == "SHORT": strength_points += 3
    if ha_today == "BULL" and ha_today != "DOJI": strength_points += 2
    elif ha_today == "BEAR" and ha_today != "DOJI": strength_points += 2
    if compression == "YES": strength_points += 2
    if rs_vs_nifty == "STRONG": strength_points += 2
    elif rs_vs_nifty == "WEAK": strength_points += 2
    if range_position in ("UPPER", "LOWER"): strength_points += 1
    if vol_ratio > 2 and breakout != "NONE": strength_points += 1
    if reversal_bonus > 0: strength_points += 1.5
    if reversal_vol_bonus > 0: strength_points += 1

    strength_tier = "S" if strength_points >= 9 else "A" if strength_points >= 7 else "B" if strength_points >= 5 else "C"

    # Final Score
    score = 0
    score += 3 if breakout == "LONG" else -3 if breakout == "SHORT" else 0
    score += ha_contribution
    score += score_add_body
    score += consecutive_doji_penalty
    score += reversal_bonus
    score += reversal_vol_bonus
    if compression == "YES" and breakout != "NONE":
        score += 1
    if BONUS_VOL_RATIO and vol_ratio > 2 and breakout != "NONE":
        score += 1
    if atr_pct < MIN_ATR_PCT:
        score *= 0.5

    return {
        "Ticker": name,
        "Final_Score": round(score, 2),
        "Final_Verdict": "STRONG BUY" if score >= 6 else "STRONG SELL" if score <= -6 else
                         "WEAK BUY" if score > 2 else "WEAK SELL" if score < -2 else "NEUTRAL",
        "Breakout": breakout,
        "Prev_HA": prev_ha,
        "HA": ha_today,
        "Trend": trend,
        "RSI": round(rsi, 1),
        "ATR%": round(atr_pct, 2),
        "Compression": compression,
        "ADX": round(float(adx), 1),
        "Vol_Ratio": round(vol_ratio, 2),
        "Price": round(price, 2),
        "Market_Regime": market_regime,
        "RS_vs_Nifty": rs_vs_nifty,
        "Range_Position": range_position,
        "Strength_Tier": strength_tier,
    }

# ==========================================================
# FUTURES SCORING
# ==========================================================
def add_futures_scoring(df):
    fut_contribution = pd.Series(0.0, index=df.index)

    def score_fut_signal(signal):
        scores = {
            "Long Buildup": 2.5,    # Stronger bullish
            "Short Covering": 2.0,
            "Long Unwinding": -2.0,
            "Short Buildup": -2.5,  # Stronger bearish
        }
        return scores.get(signal, 0)

    mask = df["F1_Signal"].notna()
    fut_contribution[mask] = df.loc[mask, "F1_Signal"].apply(score_fut_signal) * 0.7  # F1 = 70%

    oi_bonus = np.where(df["F1_OI_%"].fillna(0) > 20, 1.0,
                        np.where(df["F1_OI_%"].fillna(0) < -20, -1.0, 0))
    fut_contribution += pd.Series(oi_bonus) * 0.3  # OI = 30%

    if "F2_Signal" in df.columns:
        mask_f2 = df["F2_Signal"].notna()
        fut_contribution[mask_f2] += df.loc[mask_f2, "F2_Signal"].apply(score_fut_signal) * 0.15  # F2 = 15%

    # Divergence penalty
    cash_bull = df["Final_Score"] > 0  # Use score, not verdict (more accurate)
    fut_bull_signals = df["F1_Signal"].fillna("").isin(["Long Buildup", "Short Covering"])
    alignment_bonus = np.where(cash_bull == fut_bull_signals, 0.5, 0)
    fut_contribution += pd.Series(alignment_bonus)

    # Apply with higher weight
    futures_weight = fut_contribution * 0.5  # Now 50% influence
    df["Futures_Score"] = futures_weight.round(2)
    df["Final_Score"] += futures_weight

    # Re-evaluate verdict
    conditions = [
        df["Final_Score"] >= 6,
        df["Final_Score"] <= -6,
        df["Final_Score"] > 2,
        df["Final_Score"] < -2,
    ]
    choices = ["STRONG BUY", "STRONG SELL", "WEAK BUY", "WEAK SELL"]
    df["Final_Verdict"] = np.select(conditions, choices, default="NEUTRAL")

    return df

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
# RUN SCANNER
# ==========================================================
def run_scanner():
    symbol_map = build_ticker_universe()
    print(f"🚀 Scanner started for {len(symbol_map)} symbols...")

    nifty_df = safe_yf_download("^NSEI")
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
        print("⚠️ Nifty data unavailable — using NEUTRAL regime")
    print(f"📊 Market Regime: {market_regime}")

    results = []
    for name, sym in symbol_map.items():
        result = analyze_cash(name, sym, market_regime, nifty_return)
        if result:
            results.append(result)

    cash_df = pd.DataFrame(results)
    fut_df = run_futures(list(symbol_map.keys()))
    df = cash_df.merge(fut_df, on="Ticker", how="left")

    df = add_futures_scoring(df)

    return df.sort_values("Final_Score", ascending=False).reset_index(drop=True)

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