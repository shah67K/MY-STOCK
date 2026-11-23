import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
import os
from io import StringIO
from datetime import datetime
from zoneinfo import ZoneInfo

# ==================================================
# MY STRATEGY - CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="MY STRATEGY",
    page_icon="📈",
    layout="wide"
)

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

ST_PERIOD = 7
ST_MULTIPLIER = 2

ADX_PERIOD = 14
ADX_MIN = 20

LOOKBACK_SESSIONS = 2
BATCH_SIZE = 25

SL_ATR = 1.5
T1_ATR = 2.0
T2_ATR = 3.0

MAX_HOLDING = 10
MIN_HOLDING = 3

RISK_PERCENT = 1.0
GAP_LIMIT = 2.0

TRADE_FILE = "paper_trades.csv"

NIFTY500_URL = (
    "https://www.niftyindices.com/IndexConstituent/"
    "ind_nifty500list.csv"
)

# ==================================================
# STYLE
# ==================================================

st.markdown("""
<style>
.stApp {
    background:#101820;
    color:white;
}
h1,h2,h3 {
    color:#00d4aa;
}
div[data-testid="stMetric"] {
    background:#192630;
    padding:12px;
    border-radius:8px;
}
.stButton button {
    background:#00a884;
    color:white;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 MY STRATEGY")
st.caption(
    "NIFTY 500 | DAILY SWING SCANNER | PAPER TRADING"
)

# ==================================================
# INPUTS
# ==================================================

c1, c2 = st.columns(2)

with c1:
    capital = st.number_input(
        "Total Trading Capital (₹)",
        min_value=10000,
        max_value=100000000,
        value=200000,
        step=10000
    )

with c2:
    st.metric(
        "Maximum Risk Per Trade",
        f"₹{capital * RISK_PERCENT / 100:,.0f}"
    )

# ==================================================
# NIFTY 500 UNIVERSE
# ==================================================

@st.cache_data(ttl=3600)
def get_stocks():

    response = requests.get(
        NIFTY500_URL,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30
    )

    response.raise_for_status()

    data = pd.read_csv(
        StringIO(response.text)
    )

    col = next(
        (c for c in data.columns
         if c.strip().lower() == "symbol"),
        None
    )

    if col is None:
        raise ValueError("Symbol column not found")

    return [
        x + ".NS"
        for x in data[col].dropna()
        .astype(str).str.strip().unique()
    ]


# ==================================================
# DATA NORMALIZATION
# ==================================================

def clean_data(df):

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        return None

    df.columns = [
        str(c).strip().title()
        for c in df.columns
    ]

    required = [
        "Open", "High", "Low",
        "Close", "Volume"
    ]

    if not all(c in df.columns for c in required):
        return None

    df = df[required].copy()

    for c in required:
        df[c] = pd.to_numeric(
            df[c], errors="coerce"
        )

    df = df.dropna(subset=required)

    if df.empty:
        return None

    idx = pd.DatetimeIndex(df.index)

    if idx.tz is not None:
        idx = idx.tz_convert(
            "Asia/Kolkata"
        ).tz_localize(None)

    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    # Reject unfinished daily candle
    now = datetime.now(
        ZoneInfo("Asia/Kolkata")
    )

    if (
        df.index[-1].date() == now.date()
        and (now.hour, now.minute) < (15, 35)
    ):
        df = df.iloc[:-1]

    return df


# ==================================================
# MACD
# ==================================================

def calculate_macd(df):

    close = df["Close"]

    fast = close.ewm(
        span=MACD_FAST,
        adjust=False
    ).mean()

    slow = close.ewm(
        span=MACD_SLOW,
        adjust=False
    ).mean()

    macd = fast - slow

    signal = macd.ewm(
        span=MACD_SIGNAL,
        adjust=False
    ).mean()

    cross = (
        (macd > signal) &
        (macd.shift(1) <= signal.shift(1))
    )

    return macd, signal, cross


# ==================================================
# ATR
# ==================================================

def calculate_atr(df, period=14):

    prev = df["Close"].shift(1)

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(
        alpha=1/period,
        adjust=False,
        min_periods=period
    ).mean()


# ==================================================
# SUPERTREND
# ==================================================

def calculate_supertrend(df):

    atr = calculate_atr(
        df, ST_PERIOD
    )

    hl2 = (
        df["High"] + df["Low"]
    ) / 2

    basic_u = (
        hl2 + ST_MULTIPLIER * atr
    )

    basic_l = (
        hl2 - ST_MULTIPLIER * atr
    )

    close = df["Close"]

    upper = np.full(len(df), np.nan)
    lower = np.full(len(df), np.nan)
    trend = np.full(len(df), np.nan)

    for i in range(len(df)):

        if pd.isna(atr.iloc[i]):
            continue

        if i == 0 or np.isnan(upper[i-1]):

            upper[i] = basic_u.iloc[i]
            lower[i] = basic_l.iloc[i]
            trend[i] = 1
            continue

        pu = upper[i-1]
        pl = lower[i-1]

        upper[i] = (
            basic_u.iloc[i]
            if basic_u.iloc[i] < pu
            or close.iloc[i-1] > pu
            else pu
        )

        lower[i] = (
            basic_l.iloc[i]
            if basic_l.iloc[i] > pl
            or close.iloc[i-1] < pl
            else pl
        )

        pt = trend[i-1]

        if pt == -1 and close.iloc[i] > upper[i]:
            trend[i] = 1

        elif pt == 1 and close.iloc[i] < lower[i]:
            trend[i] = -1

        else:
            trend[i] = pt

    trend = pd.Series(
        trend, index=df.index
    )

    buy = (
        (trend == 1) &
        (trend.shift(1) == -1)
    )

    return trend, buy


# ==================================================
# ADX
# ==================================================

def calculate_adx(df):

    up = df["High"].diff()
    down = -df["Low"].diff()

    plus_dm = pd.Series(
        np.where(
            (up > down) & (up > 0),
            up, 0
        ),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (down > up) & (down > 0),
            down, 0
        ),
        index=df.index
    )

    atr = calculate_atr(
        df, ADX_PERIOD
    ).replace(0, np.nan)

    plus_sm = plus_dm.ewm(
        alpha=1/ADX_PERIOD,
        adjust=False
    ).mean()

    minus_sm = minus_dm.ewm(
        alpha=1/ADX_PERIOD,
        adjust=False
    ).mean()

    plus_di = 100 * plus_sm / atr
    minus_di = 100 * minus_sm / atr

    denom = (
        plus_di + minus_di
    ).replace(0, np.nan)

    dx = (
        100 * (plus_di - minus_di).abs()
        / denom
    )

    adx = dx.ewm(
        alpha=1/ADX_PERIOD,
        adjust=False,
        min_periods=ADX_PERIOD
    ).mean()

    return adx, plus_di, minus_di


# ==================================================
# RECENT SIGNAL AGE
# ==================================================

def signal_age(series):

    hits = np.flatnonzero(
        series.fillna(False).to_numpy()
    )

    if len(hits) == 0:
        return None

    age = len(series) - 1 - hits[-1]

    if age < LOOKBACK_SESSIONS:
        return int(age)

    return None


# ==================================================
# STOCK ANALYSIS + AUTOMATIC EXPLANATION
# ==================================================

def analyse_stock(ticker, raw):

    df = clean_data(raw)

    if df is None or len(df) < 100:
        return None

    macd, signal, cross = calculate_macd(df)

    st_trend, st_buy = calculate_supertrend(df)

    adx, plus, minus = calculate_adx(df)

    m_age = signal_age(cross)
    s_age = signal_age(st_buy)

    if m_age is None or s_age is None:
        return None

    last_adx = adx.iloc[-1]
    prev_adx = adx.iloc[-2]
    last_plus = plus.iloc[-1]
    last_minus = minus.iloc[-1]

    if any(pd.isna(x) for x in [
        last_adx, prev_adx,
        last_plus, last_minus
    ]):
        return None

    if last_adx < ADX_MIN:
        return None

    if last_plus <= last_minus:
        return None

    if st_trend.iloc[-1] != 1:
        return None

    atr = calculate_atr(
        df, 14
    ).iloc[-1]

    if pd.isna(atr) or atr <= 0:
        return None

    close = float(df["Close"].iloc[-1])
    previous_high = float(df["High"].iloc[-1])

    adx_rising = bool(last_adx > prev_adx)

    di_spread = float(last_plus - last_minus)

    gap_to_trigger = (
        (previous_high - close) / close
    ) * 100

    # ------------------------------------------
    # SIGNAL FRESHNESS
    # ------------------------------------------

    macd_text = (
        "Fresh bullish crossover"
        if m_age == 0
        else "MACD crossover 1 session old"
    )

    st_text = (
        "Fresh Supertrend BUY"
        if s_age == 0
        else "Supertrend BUY 1 session old"
    )

    # ------------------------------------------
    # ADX ANALYSIS
    # ------------------------------------------

    if adx_rising:

        adx_text = (
            f"ADX rising ({last_adx:.2f}) - "
            "trend strength improving"
        )

    else:

        adx_text = (
            f"ADX not rising ({last_adx:.2f}) - "
            "trend strength is not improving"
        )

    # ------------------------------------------
    # DI ANALYSIS
    # ------------------------------------------

    if di_spread >= 8:

        di_text = (
            f"+DI clearly above -DI "
            f"(difference {di_spread:.2f})"
        )

    elif di_spread >= 4:

        di_text = (
            f"+DI moderately above -DI "
            f"(difference {di_spread:.2f})"
        )

    else:

        di_text = (
            f"+DI and -DI very close "
            f"(difference {di_spread:.2f})"
        )

    # ------------------------------------------
    # ENTRY DISTANCE ANALYSIS
    # ------------------------------------------

    if gap_to_trigger > 2:

        entry_text = (
            f"Entry trigger is {gap_to_trigger:.2f}% "
            "above current close - relatively far"
        )

    elif gap_to_trigger > 1:

        entry_text = (
            f"Entry trigger is {gap_to_trigger:.2f}% "
            "above current close"
        )

    else:

        entry_text = (
            f"Entry trigger is only {gap_to_trigger:.2f}% "
            "above current close"
        )

    # ------------------------------------------
    # SETUP QUALITY SCORE
    # ------------------------------------------

    score = 0

    score += 2 if m_age == 0 else 1
    score += 2 if s_age == 0 else 1

    score += 1 if adx_rising else 0
    score += 1 if last_adx >= 25 else 0

    score += 1 if di_spread >= 8 else 0

    score += 1 if gap_to_trigger <= 2 else 0

    if score >= 6:
        quality = "STRONG"

    elif score >= 4:
        quality = "MODERATE"

    else:
        quality = "WEAK"

    # ------------------------------------------
    # FINAL VIEW
    # ------------------------------------------

    if quality == "WEAK":

        final_view = "AVOID / WEAK SETUP"

    elif quality == "MODERATE":

        final_view = "WAIT - CONFIRM BREAKOUT"

    else:

        final_view = (
            "BUY CANDIDATE - WAIT FOR BREAKOUT"
        )

    # ------------------------------------------
    # TECHNICAL ANALYSIS TEXT
    # ------------------------------------------

    analysis = "\n".join([
        f"MACD: {macd_text}",
        f"Supertrend: {st_text}",
        f"ADX: {adx_text}",
        f"DI: {di_text}",
        f"Entry: {entry_text}",
        f"Setup Quality: {quality}",
        f"Final View: {final_view}"
    ])

    signal_date = df.index[-1]

    return {
        "Stock": ticker.replace(".NS", ""),
        "Signal Date": signal_date.strftime("%Y-%m-%d"),

        "Close": round(close, 2),
        "Previous High": round(previous_high, 2),
        "ATR": round(float(atr), 2),

        "MACD Age": m_age,
        "ST Age": s_age,

        "ADX": round(float(last_adx), 2),
        "+DI": round(float(last_plus), 2),
        "-DI": round(float(last_minus), 2),

        "ADX Rising": adx_rising,
        "DI Spread": round(di_spread, 2),
        "Gap to Trigger %": round(gap_to_trigger, 2),

        "Setup Quality": quality,
        "Final View": final_view,
        "Technical Analysis": analysis
    }


# ==================================================
# PAPER TRADE FILE
# ==================================================

TRADE_COLUMNS = [
    "Stock", "Signal Date", "Buy Above", "ATR",
    "Risk Budget", "Status", "Entry Date",
    "Entry", "SL", "T1", "T2", "Quantity",
    "T1 Quantity", "Remaining Quantity",
    "T1 Booked", "Realized PnL",
    "Exit Date", "Exit Price", "Holding Days",
    "Exit Reason"
]


def load_trades():

    if os.path.exists(TRADE_FILE):

        try:

            df = pd.read_csv(TRADE_FILE)

            for c in TRADE_COLUMNS:
                if c not in df.columns:
                    df[c] = np.nan

            return df[TRADE_COLUMNS]

        except Exception:

            st.error("Paper trade CSV read error.")

    return pd.DataFrame(
        columns=TRADE_COLUMNS
    )


def save_trades(df):

    df.to_csv(
        TRADE_FILE,
        index=False
    )


# ==================================================
# CREATE NEW PAPER SETUPS
# ==================================================

def add_new_setups(trades, signals, capital):

    for s in signals:

        stock = s["Stock"]

        active = trades[
            (trades["Stock"] == stock)
            & trades["Status"].isin(
                ["PENDING", "OPEN", "T1 HIT"]
            )
        ]

        if not active.empty:
            continue

        exists = trades[
            (trades["Stock"] == stock)
            & (trades["Signal Date"] == s["Signal Date"])
        ]

        if not exists.empty:
            continue

        row = {
            "Stock": stock,
            "Signal Date": s["Signal Date"],
            "Buy Above": s["Previous High"],
            "ATR": s["ATR"],
            "Risk Budget": capital * RISK_PERCENT / 100,
            "Status": "PENDING",
            "Entry Date": "",
            "Entry": np.nan,
            "SL": np.nan,
            "T1": np.nan,
            "T2": np.nan,
            "Quantity": 0,
            "T1 Quantity": 0,
            "Remaining Quantity": 0,
            "T1 Booked": False,
            "Realized PnL": 0.0,
            "Exit Date": "",
            "Exit Price": np.nan,
            "Holding Days": 0,
            "Exit Reason": ""
        }

        trades = pd.concat(
            [trades, pd.DataFrame([row])],
            ignore_index=True
        )

    return trades


# ==================================================
# REPLAY PAPER TRADE FROM DAILY CANDLES
# ==================================================

def replay_trade(trade, df):

    trade = trade.copy()

    signal_date = pd.Timestamp(
        trade["Signal Date"]
    )

    bars = df[
        df.index > signal_date
    ]

    trade["Status"] = "PENDING"
    trade["Entry Date"] = ""
    trade["Entry"] = np.nan
    trade["SL"] = np.nan
    trade["T1"] = np.nan
    trade["T2"] = np.nan
    trade["Quantity"] = 0
    trade["T1 Quantity"] = 0
    trade["Remaining Quantity"] = 0
    trade["T1 Booked"] = False
    trade["Realized PnL"] = 0.0
    trade["Exit Date"] = ""
    trade["Exit Price"] = np.nan
    trade["Holding Days"] = 0
    trade["Exit Reason"] = ""

    buy_above = float(trade["Buy Above"])
    atr = float(trade["ATR"])
    risk_budget = float(trade["Risk Budget"])

    if bars.empty:
        return trade

    entry = None
    qty = 0
    remaining = 0
    t1_qty = 0
    booked = 0.0
    t1_hit = False
    holding = 0

    sl = t1 = t2 = np.nan

    for date, bar in bars.iterrows():

        op = float(bar["Open"])
        hi = float(bar["High"])
        lo = float(bar["Low"])
        cl = float(bar["Close"])

        # WAITING FOR ENTRY

        if entry is None:

            if op > buy_above * (1 + GAP_LIMIT / 100):

                trade["Status"] = "SKIPPED GAP"
                trade["Exit Date"] = str(date.date())
                trade["Exit Reason"] = "Gap-up above 2%"
                break

            if hi < buy_above:
                continue

            entry = max(op, buy_above)

            sl = entry - SL_ATR * atr
            t1 = entry + T1_ATR * atr
            t2 = entry + T2_ATR * atr

            if sl <= 0:

                trade["Status"] = "SKIPPED"
                trade["Exit Reason"] = "Invalid SL"
                break

            qty = int(
                risk_budget / (entry - sl)
            )

            if qty < 1:

                trade["Status"] = "SKIPPED"
                trade["Exit Reason"] = (
                    "Risk budget insufficient for 1 share"
                )
                break

            t1_qty = qty // 2
            remaining = qty
            holding = 1

            trade["Entry"] = round(entry, 2)
            trade["Entry Date"] = str(date.date())
            trade["SL"] = round(sl, 2)
            trade["T1"] = round(t1, 2)
            trade["T2"] = round(t2, 2)
            trade["Quantity"] = qty
            trade["T1 Quantity"] = t1_qty

        else:

            holding += 1

        # ACTIVE TRADE

        if not t1_hit:

            if op <= sl:

                exit_price = op

                booked += remaining * (
                    exit_price - entry
                )

                remaining = 0
                trade["Status"] = "SL HIT"
                trade["Exit Reason"] = "Gap below SL"

            elif lo <= sl:

                booked += remaining * (
                    sl - entry
                )

                remaining = 0
                trade["Status"] = "SL HIT"
                trade["Exit Reason"] = "Stop Loss"

            elif hi >= t1:

                booked += t1_qty * (
                    t1 - entry
                )

                remaining -= t1_qty
                t1_hit = True

                sl = entry

                if hi >= t2 and remaining > 0:

                    booked += remaining * (
                        t2 - entry
                    )

                    remaining = 0
                    trade["Status"] = "T2 HIT"
                    trade["Exit Reason"] = "Target 2"

                else:

                    trade["Status"] = "T1 HIT"

        else:

            if op <= entry:

                booked += remaining * (
                    op - entry
                )

                remaining = 0
                trade["Status"] = "BREAKEVEN EXIT"
                trade["Exit Reason"] = (
                    "Gap below breakeven"
                )

            elif lo <= entry:

                remaining = 0
                trade["Status"] = "BREAKEVEN EXIT"
                trade["Exit Reason"] = "Breakeven SL"

            elif hi >= t2:

                booked += remaining * (
                    t2 - entry
                )

                remaining = 0
                trade["Status"] = "T2 HIT"
                trade["Exit Reason"] = "Target 2"

        trade["Realized PnL"] = round(
            booked, 2
        )

        trade["Remaining Quantity"] = remaining
        trade["T1 Booked"] = t1_hit
        trade["Holding Days"] = holding

        # EXIT AFTER 10 SESSIONS

        if remaining > 0 and holding >= MAX_HOLDING:

            booked += remaining * (
                cl - entry
            )

            remaining = 0
            trade["Status"] = "TIME EXIT"
            trade["Exit Reason"] = "10 session limit"

        if remaining == 0:

            trade["Exit Date"] = str(date.date())

            trade["Exit Price"] = round(
                op if (
                    trade["Exit Reason"] == "Gap below SL"
                    or trade["Exit Reason"] == "Gap below breakeven"
                ) else (
                    sl if trade["Exit Reason"] == "Stop Loss"
                    else entry if trade["Exit Reason"] == "Breakeven SL"
                    else t2 if trade["Exit Reason"] == "Target 2"
                    else cl
                ), 2
            )

            trade["Realized PnL"] = round(
                booked, 2
            )

            trade["Remaining Quantity"] = 0
            break

    # STILL OPEN

    if entry is not None and remaining > 0:

        trade["Remaining Quantity"] = remaining

        trade["Realized PnL"] = round(
            booked, 2
        )

        trade["Status"] = (
            "T1 HIT" if t1_hit else "OPEN"
        )

    return trade


# ==================================================
# YFINANCE BATCH EXTRACTION
# ==================================================

def extract_data(data, ticker, batch_size):

    if data is None or data.empty:
        return None

    if not isinstance(data.columns, pd.MultiIndex):

        if batch_size == 1:
            return data.copy()

        return None

    for level in range(data.columns.nlevels):

        if ticker in data.columns.get_level_values(level):

            return data.xs(
                ticker,
                axis=1,
                level=level
            ).copy()

    return None


# ==================================================
# MAIN SCAN
# ==================================================

def run_scanner(capital):

    stocks = get_stocks()

    signals = []
    market_data = {}

    progress = st.progress(0)
    status = st.empty()

    total = len(stocks)

    for start in range(0, total, BATCH_SIZE):

        batch = stocks[
            start:start+BATCH_SIZE
        ]

        data = yf.download(
            batch,
            period="6mo",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            threads=False,
            progress=False,
            multi_level_index=True
        )

        if data is not None and not data.empty:

            for ticker in batch:

                raw = extract_data(
                    data,
                    ticker,
                    len(batch)
                )

                if raw is None:
                    continue

                df = clean_data(raw)

                if df is None:
                    continue

                market_data[
                    ticker.replace(".NS", "")
                ] = df

                result = analyse_stock(
                    ticker, raw
                )

                if result is not None:
                    signals.append(result)

        done = min(
            start + BATCH_SIZE, total
        )

        progress.progress(done / total)

        status.info(
            f"Scanned {done}/{total} | "
            f"Candidates: {len(signals)}"
        )

        time.sleep(0.5)

    progress.empty()
    status.empty()

    trades = load_trades()

    trades = add_new_setups(
        trades,
        signals,
        capital
    )

    for i in trades.index:

        stock = str(trades.at[i, "Stock"])

        if stock not in market_data:
            continue

        updated = replay_trade(
            trades.loc[i],
            market_data[stock]
        )

        for col in TRADE_COLUMNS:
            trades.at[i, col] = updated[col]

    save_trades(trades)

    return signals, trades


# ==================================================
# DASHBOARD OVERVIEW
# ==================================================

st.markdown("---")

a, b, c = st.columns(3)

a.metric("TIMEFRAME", "DAILY")
b.metric("HOLDING", "3–10 DAYS")
c.metric("RISK", "1% / TRADE")

st.markdown("""
**ENTRY FILTER**

- MACD bullish crossover: Last 2 completed sessions
- Supertrend fresh BUY: Last 2 completed sessions
- ADX ≥ 20
- +DI > -DI
- Supertrend currently bullish

**EXIT PLAN**

- SL = 1.5 ATR
- T1 = 2 ATR (50% quantity)
- T2 = 3 ATR (remaining quantity)
- After T1: SL moves to breakeven
- Maximum holding = 10 sessions
""")

# ==================================================
# SCAN BUTTON
# ==================================================

if st.button(
    "🔍 SCAN NIFTY 500 & UPDATE TRADES",
    use_container_width=True
):

    try:

        with st.spinner(
            "Scanning market and updating paper trades..."
        ):

            signals, trades = run_scanner(
                capital
            )

        st.session_state["signals"] = signals
        st.session_state["trades"] = trades

        st.session_state["scan_time"] = (
            datetime.now(
                ZoneInfo("Asia/Kolkata")
            ).strftime("%d-%m-%Y %H:%M:%S")
        )

        st.success(
            "Scan and paper trade update complete."
        )

    except Exception as e:

        st.error(f"Scanner error: {e}")


# ==================================================
# NEW BUY CANDIDATES + ANALYSIS
# ==================================================

if "signals" in st.session_state:

    st.markdown("---")
    st.subheader("🎯 NEW BUY CANDIDATES")

    signals = st.session_state["signals"]

    st.caption(
        "Last scan: " +
        st.session_state["scan_time"]
    )

    if signals:

        signal_df = pd.DataFrame(signals)

        quality_order = {
            "STRONG": 0,
            "MODERATE": 1,
            "WEAK": 2
        }

        signal_df["_quality_order"] = (
            signal_df["Setup Quality"]
            .map(quality_order)
        )

        signal_df = signal_df.sort_values(
            [
                "_quality_order",
                "MACD Age",
                "ST Age",
                "ADX"
            ],
            ascending=[True, True, True, False]
        )

        # --------------------------------------
        # SUMMARY TABLE
        # --------------------------------------

        st.markdown("### 📋 CANDIDATE SUMMARY")

        display_columns = [
            "Stock",
            "Close",
            "Previous High",
            "ATR",
            "MACD Age",
            "ST Age",
            "ADX",
            "+DI",
            "-DI",
            "ADX Rising",
            "DI Spread",
            "Gap to Trigger %",
            "Setup Quality",
            "Final View"
        ]

        st.dataframe(
            signal_df[display_columns],
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------
        # DETAILED ANALYSIS
        # --------------------------------------

        st.markdown("---")
        st.subheader("🧠 DETAILED TECHNICAL ANALYSIS")

        for _, row in signal_df.iterrows():

            quality = row["Setup Quality"]

            if quality == "STRONG":
                icon = "🟢"

            elif quality == "MODERATE":
                icon = "🟡"

            else:
                icon = "🔴"

            with st.expander(
                f"{icon} {row['Stock']} | "
                f"{quality} | {row['Final View']}",
                expanded=True
            ):

                left, right = st.columns(2)

                with left:

                    st.markdown("**SIGNAL STATUS**")

                    st.write(
                        "• MACD: " +
                        (
                            "Fresh bullish crossover"
                            if row["MACD Age"] == 0
                            else "Crossover 1 session old"
                        )
                    )

                    st.write(
                        "• Supertrend: " +
                        (
                            "Fresh BUY signal"
                            if row["ST Age"] == 0
                            else "BUY signal 1 session old"
                        )
                    )

                    st.write(
                        "• ADX: " +
                        (
                            "Rising"
                            if row["ADX Rising"]
                            else "Not rising"
                        )
                    )

                with right:

                    st.markdown("**STRENGTH & ENTRY**")

                    st.write(
                        f"• ADX Value: {row['ADX']}"
                    )

                    st.write(
                        f"• +DI / -DI: "
                        f"{row['+DI']} / {row['-DI']}"
                    )

                    st.write(
                        f"• DI Difference: "
                        f"{row['DI Spread']}"
                    )

                    st.write(
                        f"• Entry trigger distance: "
                        f"{row['Gap to Trigger %']}%"
                    )

                st.markdown("**AUTOMATIC ANALYSIS**")

                st.text(
                    row["Technical Analysis"]
                )

                st.markdown("**TRADE LEVELS**")

                entry = float(row["Previous High"])
                atr_value = float(row["ATR"])

                sl = entry - SL_ATR * atr_value
                t1 = entry + T1_ATR * atr_value
                t2 = entry + T2_ATR * atr_value

                p1, p2, p3, p4 = st.columns(4)

                p1.metric(
                    "BUY ABOVE",
                    f"₹{entry:,.2f}"
                )

                p2.metric(
                    "STOP LOSS",
                    f"₹{sl:,.2f}"
                )

                p3.metric(
                    "TARGET 1",
                    f"₹{t1:,.2f}"
                )

                p4.metric(
                    "TARGET 2",
                    f"₹{t2:,.2f}"
                )

                st.info(
                    "Final decision: "
                    + str(row["Final View"])
                    + ". Breakout confirmation ke bina entry nahi leni."
                )

    else:

        st.info(
            "No matching stocks today."
        )


# ==================================================
# PAPER TRADING BOOK
# ==================================================

st.markdown("---")
st.subheader("📒 PAPER TRADING BOOK")

trades = load_trades()

if not trades.empty:

    active = trades[
        trades["Status"].isin(
            ["PENDING", "OPEN", "T1 HIT"]
        )
    ].copy()

    completed = trades[
        ~trades["Status"].isin(
            ["PENDING", "OPEN", "T1 HIT"]
        )
    ].copy()

    x, y, z = st.columns(3)

    x.metric(
        "ACTIVE / PENDING",
        len(active)
    )

    y.metric(
        "COMPLETED",
        len(completed)
    )

    pnl = pd.to_numeric(
        trades["Realized PnL"],
        errors="coerce"
    ).fillna(0).sum()

    z.metric(
        "REALIZED P&L",
        f"₹{pnl:,.2f}",
        delta=f"{pnl:,.2f}"
    )

    st.markdown("**ACTIVE TRADES**")

    if not active.empty:

        st.dataframe(
            active[
                [
                    "Stock", "Status", "Buy Above",
                    "Entry", "SL", "T1", "T2",
                    "Quantity", "Remaining Quantity",
                    "Holding Days", "Realized PnL"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info("No active trades.")

    st.markdown("**TRADE HISTORY**")

    if not completed.empty:

        st.dataframe(
            completed[
                [
                    "Stock", "Signal Date",
                    "Entry Date", "Entry",
                    "Exit Date", "Exit Price",
                    "Exit Reason", "Holding Days",
                    "Realized PnL"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    csv = trades.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ DOWNLOAD PAPER TRADE HISTORY",
        csv,
        "MY_STRATEGY_PAPER_TRADES.csv",
        "text/csv"
    )

else:

    st.info(
        "Abhi koi paper trade record nahi hai. "
        "Pehle scanner run karo."
    )


# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "MY STRATEGY | Daily swing trading | "
    "Paper trading only | Not a live buy recommendation"
)
