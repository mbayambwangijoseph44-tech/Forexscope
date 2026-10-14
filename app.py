import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# =========================================================
# 1. CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="ForexScope",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = "forexscope.db"

SYMBOLS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X",
    "USD/CHF": "CHF=X",
    "NZD/USD": "NZDUSD=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "Or (Gold)": "GC=F",
    "Argent (Silver)": "SI=F",
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Bitcoin / USD": "BTC-USD",
}

INTERVALS = {
    "M1": {"interval": "1m", "period": "7d"},
    "M5": {"interval": "5m", "period": "60d"},
    "M15": {"interval": "15m", "period": "60d"},
    "M30": {"interval": "30m", "period": "60d"},
    "H1": {"interval": "60m", "period": "60d"},
    "H4": {"interval": "60m", "period": "60d"},
    "D1": {"interval": "1d", "period": "2y"},
}

# Délai minimal entre deux signaux actionnables sur le même symbole
# et la même unité de temps. Valeurs initiales configurables dans la sidebar.
DEFAULT_SIGNAL_COOLDOWN_MINUTES = {
    "M1": 5,
    "M5": 15,
    "M15": 30,
    "M30": 60,
    "H1": 120,
    "H4": 480,
    "D1": 1440,
}

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }
        [data-testid="stMetric"] {
            background: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.2);
            padding: 16px;
            border-radius: 12px;
        }
        div[data-testid="stTabs"] button {
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 2. DATA VALIDATION AND RETRIEVAL
# =========================================================
def validate_ohlc(data):
    """Clean and validate OHLC data."""
    if data is None or data.empty:
        return pd.DataFrame()

    data = data.copy()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns = [str(c).strip().lower() for c in data.columns]
    required = ["open", "high", "low", "close"]

    if not all(c in data.columns for c in required):
        return pd.DataFrame()

    if "time" not in data.columns:
        data["time"] = data.index

    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["time"] = pd.to_datetime(data["time"], errors="coerce", utc=True)
    data = data.dropna(subset=["time", "open", "high", "low", "close"])

    data = data[
        (data["high"] >= data["low"])
        & (data["high"] >= data["open"])
        & (data["high"] >= data["close"])
        & (data["low"] <= data["open"])
        & (data["low"] <= data["close"])
        & (data["close"] > 0)
    ]

    data = (
        data.sort_values("time")
        .drop_duplicates(subset=["time"])
        .reset_index(drop=True)
    )
    return data[["time", "open", "high", "low", "close"]]


@st.cache_data(ttl=30, show_spinner=False)
def fetch_candles(symbol, timeframe, count=300):
    """Download candles and return validated OHLC data."""
    if timeframe not in INTERVALS:
        return pd.DataFrame()

    settings = INTERVALS[timeframe]

    try:
        data = yf.download(
            tickers=symbol,
            period=settings["period"],
            interval=settings["interval"],
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="column",
        )

        if data is None or data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = validate_ohlc(data)
        if data.empty:
            return pd.DataFrame()

        # Build H4 candles from H1 candles.
        if timeframe == "H4":
            data = data.set_index("time")
            data = data.resample(
                "4h",
                origin="start_day",
                label="left",
                closed="left",
            ).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
            })
            data = data.dropna().reset_index()
            data = validate_ohlc(data)

        return data.tail(int(count)).reset_index(drop=True)

    except Exception:
        return pd.DataFrame()


# =========================================================
# 3. TECHNICAL INDICATORS
# =========================================================
def calculate_indicators(data):
    if data is None or data.empty:
        return pd.DataFrame()

    df = data.copy()
    close, high, low = df["close"], df["high"], df["low"]

    df["ema_20"] = close.ewm(span=20, adjust=False).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False).mean()
    df["ema_200"] = close.ewm(span=200, adjust=False).mean()

    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df.loc[avg_loss == 0, "rsi"] = 100

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = true_range.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )
    atr_safe = df["atr"].replace(0, np.nan)
    plus_di = (
        100
        * plus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        / atr_safe
    )
    minus_di = (
        100
        * minus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        / atr_safe
    )
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["adx"] = dx.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    return df


def analyze_market_structure(data, lookback=20):
    """Describe a simple close-confirmed break of the prior range.

    The caller should pass only candles considered closed. This is a practical
    BOS proxy, not a full discretionary swing-point/CHOCH implementation.
    """
    result = {
        "structure": "Données insuffisantes",
        "bias": "Neutre",
        "last_high": None,
        "last_low": None,
        "breakout": None,
    }
    if data is None or len(data) < max(lookback + 1, 10):
        return result

    recent = data.tail(lookback + 1).copy()
    previous = recent.iloc[:-1]
    current = recent.iloc[-1]
    if previous.empty:
        return result

    prior_high = float(previous["high"].max())
    prior_low = float(previous["low"].min())
    close = float(current["close"])
    result["last_high"] = prior_high
    result["last_low"] = prior_low

    if close > prior_high:
        result.update(structure="Cassure haussière (BOS potentiel)", bias="Haussier", breakout="HAUSSIER")
    elif close < prior_low:
        result.update(structure="Cassure baissière (BOS potentiel)", bias="Baissier", breakout="BAISSIER")
    else:
        first_close = float(previous["close"].iloc[0])
        if close > first_close:
            result.update(structure="Progression haussière sans cassure", bias="Haussier")
        elif close < first_close:
            result.update(structure="Progression baissière sans cassure", bias="Baissier")
        else:
            result.update(structure="Range / neutre", bias="Neutre")
    return result


def detect_liquidity_sweep(data, lookback=20):
    """Detect a sweep on the final candle of the supplied (closed) dataset."""
    result = {"sweep": "Aucun balayage détecté", "level": None, "extreme": None, "index": None}
    if data is None or len(data) < lookback + 1:
        return result

    previous = data.iloc[-(lookback + 1):-1]
    last = data.iloc[-1]
    previous_low = float(previous["low"].min())
    previous_high = float(previous["high"].max())
    low = float(last["low"])
    high = float(last["high"])
    close = float(last["close"])

    if low < previous_low and close > previous_low:
        result.update(sweep="Sweep haussier potentiel", level=previous_low, extreme=low, index=len(data) - 1)
    elif high > previous_high and close < previous_high:
        result.update(sweep="Sweep baissier potentiel", level=previous_high, extreme=high, index=len(data) - 1)
    return result


def _is_impulsive_candle(candle, atr, direction):
    """Require directional body size and a decisive close within the range."""
    if not np.isfinite(atr) or atr <= 0:
        return False
    open_price = float(candle["open"])
    high = float(candle["high"])
    low = float(candle["low"])
    close = float(candle["close"])
    candle_range = high - low
    body = abs(close - open_price)
    if candle_range <= 0 or body < 0.65 * atr or body / candle_range < 0.55:
        return False
    if direction == "Haussier":
        return close > open_price and (high - close) / candle_range <= 0.25
    return close < open_price and (close - low) / candle_range <= 0.25


def _find_confirmed_smc_setup(closed, atr, lookback=20, sweep_window=6):
    """Find sweep -> later structure break -> impulse using closed candles only."""
    empty = {"direction": None, "sweep_extreme": None, "reasons": []}
    if closed is None or len(closed) < lookback + 3:
        return empty
        # The final candle is the potential confirmation candle. A sweep must have
    # happened earlier; a sweep alone is never an entry signal.
    current = closed.iloc[-1]
    current_index = len(closed) - 1
    first_sweep_index = max(lookback, current_index - sweep_window)

    for sweep_idx in range(current_index - 1, first_sweep_index - 1, -1):
        sweep_candle = closed.iloc[sweep_idx]
        prior = closed.iloc[max(0, sweep_idx - lookback):sweep_idx]
        if len(prior) < min(lookback, 10):
            continue
        prior_low = float(prior["low"].min())
        prior_high = float(prior["high"].max())
        sweep_low = float(sweep_candle["low"])
        sweep_high = float(sweep_candle["high"])
        sweep_close = float(sweep_candle["close"])

        # Require a rejection through an earlier liquidity level.
        if sweep_low < prior_low and sweep_close > prior_low:
            direction = "Haussier"
            extreme = sweep_low
            later = closed.iloc[sweep_idx + 1:current_index]
            reference = float(later["high"].max()) if not later.empty else float(prior["high"].tail(5).max())
            broken = float(current["close"]) > reference
        elif sweep_high > prior_high and sweep_close < prior_high:
            direction = "Baissier"
            extreme = sweep_high
            later = closed.iloc[sweep_idx + 1:current_index]
            reference = float(later["low"].min()) if not later.empty else float(prior["low"].tail(5).min())
            broken = float(current["close"]) < reference
        else:
            continue

        if broken and _is_impulsive_candle(current, atr, direction):
            return {
                "direction": direction,
                "sweep_extreme": extreme,
                "reasons": [
                    f"Sweep {direction.lower()} détecté sur une bougie clôturée.",
                    f"Cassure de structure {direction.lower()} après le sweep.",
                    "Bougie de confirmation impulsive.",
                ],
            }
    return empty

def analyze_timeframe(raw_data):
    insufficient = {
        "status": "Données insuffisantes",
        "bias": "Neutre",
        "rsi": None,
        "adx": None,
        "atr": None,
        "macd": "Indisponible",
        "structure": "Indisponible",
        "sweep": "Indisponible",
        "close": None,
        "candle_time": None,
    }

    if raw_data is None or len(raw_data) < 50:
        return insufficient

    df = calculate_indicators(raw_data)
    df = df.dropna(subset=["ema_20", "ema_50", "rsi", "atr"])
    if df.empty:
        return {**insufficient, "status": "Indicateurs indisponibles"}

    closed = df.iloc[:-1].copy() if len(df) >= 2 else df.copy()
    last = closed.iloc[-1]
    bullish_points = 0
    bearish_points = 0

    if last["close"] > last["ema_20"]:
        bullish_points += 1
    elif last["close"] < last["ema_20"]:
        bearish_points += 1

    if last["ema_20"] > last["ema_50"]:
        bullish_points += 1
    elif last["ema_20"] < last["ema_50"]:
        bearish_points += 1

    if last["macd"] > last["macd_signal"]:
        bullish_points += 1
    elif last["macd"] < last["macd_signal"]:
        bearish_points += 1

    if last["rsi"] > 55:
        bullish_points += 1
    elif last["rsi"] < 45:
        bearish_points += 1

    if bullish_points > bearish_points:
        bias = "Haussier"
    elif bearish_points > bullish_points:
        bias = "Baissier"
    else:
        bias = "Neutre"

    structure = analyze_market_structure(closed)
    sweep = detect_liquidity_sweep(closed)

    if last["macd"] > last["macd_signal"]:
        macd_state = "Haussier"
    elif last["macd"] < last["macd_signal"]:
        macd_state = "Baissier"
    else:
        macd_state = "Neutre"

    return {
        "status": "OK",
        "bias": bias,
        "rsi": float(last["rsi"]),
        "adx": float(last["adx"]) if pd.notna(last["adx"]) else None,
        "atr": float(last["atr"]),
        "macd": macd_state,
        "structure": structure["structure"],
        "sweep": sweep["sweep"],
        "close": float(last["close"]),
        "ema_20": float(last["ema_20"]),
        "ema_50": float(last["ema_50"]),
        "ema_200": float(last["ema_200"]) if pd.notna(last["ema_200"]) else None,
        "bullish_points": bullish_points,
        "bearish_points": bearish_points,
        "candle_time": last["time"],
    }


def analyze_multiple_timeframes(symbol, count=300):
    results = {}
    for tf in ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]:
        raw_data = fetch_candles(symbol, tf, count)
        if raw_data.empty:
            results[tf] = {
                "status": "Données indisponibles",
                "bias": "Neutre",
                "rsi": None,
                "adx": None,
                "atr": None,
                "macd": "Indisponible",
                "structure": "Indisponible",
                "sweep": "Indisponible",
                "close": None,
                "candle_time": None,
            }
        else:
            results[tf] = analyze_timeframe(raw_data)
    return results


def assess_smc_zone_confluence(closed_data, direction, price, atr):
    """Assess whether price is near a directional FVG/order-block candidate.

    This is supporting context only: candidate zones never create a trade by
    themselves and never override the sweep/BOS/impulse and H4/D1 filters.
    """
    result = {"score_adjustment": 0, "nearby_zones": [], "note": "Aucune zone SMC proche confirmée."}
    if closed_data is None or closed_data.empty or not np.isfinite(price) or not np.isfinite(atr) or atr <= 0:
        return result

    wanted = "Haussier" if direction == "Haussier" else "Baissier"
    candidates = []
    for zone in detect_fair_value_gaps(closed_data, max_zones=8) + detect_order_blocks(closed_data, max_zones=6):
        if zone.get("kind") != wanted:
            continue
        low, high = float(zone["low"]), float(zone["high"])
        distance = 0.0 if low <= price <= high else min(abs(price-low), abs(price-high))
        if distance <= 0.75 * atr:
            candidates.append({"type": "FVG" if "FVG" in str(zone.get("label", "")) else "Zone SMC", "kind": wanted,
                               "low": low, "high": high, "distance_atr": distance/atr})
    if candidates:
        candidates.sort(key=lambda z: z["distance_atr"])
        best = candidates[0]
        result.update(score_adjustment=5, nearby_zones=candidates[:3],
                      note=f"Zone SMC candidate {best['low']:.5f}–{best['high']:.5f} proche du prix ({best['distance_atr']:.2f} ATR).")
    else:
        result["score_adjustment"] = -5
        result["note"] = "Aucune FVG/zone OB candidate de même direction à moins de 0,75 ATR."
    return result


# =========================================================
# 4. SIGNAL ENGINE
# =========================================================
def generate_trading_signal(raw_data, multi_results):
    """Generate a conservative signal only after sweep, BOS and impulse align."""
    neutral_result = {
        "signal": "ATTENDRE",
        "confidence": 0,  # Legacy DB field; displayed as setup quality, not win probability.
        "setup_quality": 0,
        "confirmation": "Non confirmée",
        "bias": "Neutre",
        "entry": None,
        "stop_loss": None,
        "take_profit": None,
        "risk_reward": None,
        "reasons": ["Données insuffisantes pour décider."],
        "rsi": None,
        "atr": None,
        "bullish_score": 0,
        "bearish_score": 0,
        "candle_time": None,
    }
    if raw_data is None or len(raw_data) < 60:
        return neutral_result

    df = calculate_indicators(raw_data)
    df = df.dropna(subset=["ema_20", "ema_50", "rsi", "atr"])
    # Exclude the newest row conservatively because Yahoo may return a live candle.
    closed = df.iloc[:-1].copy()
    if len(closed) < 50:
        return neutral_result

    last = closed.iloc[-1]
    close = float(last["close"])
    atr = float(last["atr"])
    rsi = float(last["rsi"])
    candle_time = last["time"]
    if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
        return {**neutral_result, "candle_time": candle_time}

    setup = _find_confirmed_smc_setup(closed, atr)
    reasons = list(setup["reasons"])
    if setup["direction"] is None:
        reasons.append("Sweep, cassure de structure et impulsion ne sont pas réunis dans cet ordre.")
        return {**neutral_result, "reasons": reasons, "candle_time": candle_time, "rsi": rsi, "atr": atr}

    direction = setup["direction"]
    # Higher-timeframe context is a filter, not a point added to a loose score.
    h4 = multi_results.get("H4", {}) if isinstance(multi_results, dict) else {}
    d1 = multi_results.get("D1", {}) if isinstance(multi_results, dict) else {}
    h4_bias = h4.get("bias") if h4.get("status") == "OK" else None
    d1_bias = d1.get("bias") if d1.get("status") == "OK" else None
    higher_biases = [b for b in (h4_bias, d1_bias) if b in ("Haussier", "Baissier")]

    if len(higher_biases) == 2 and higher_biases[0] != higher_biases[1]:
        reasons.append("H4 et D1 sont contradictoires : signal bloqué.")
        return {**neutral_result, "reasons": reasons, "candle_time": candle_time, "rsi": rsi, "atr": atr, "bias": direction}
    if higher_biases and any(b != direction for b in higher_biases):
        reasons.append("Le setup contredit le biais H4/D1 : confirmation insuffisante.")
        return {**neutral_result, "reasons": reasons, "candle_time": candle_time, "rsi": rsi, "atr": atr, "bias": direction}
    if not higher_biases:
        reasons.append("Biais H4/D1 indisponible : confirmation supérieure absente.")
        return {**neutral_result, "reasons": reasons, "candle_time": candle_time, "rsi": rsi, "atr": atr, "bias": direction}

    # Momentum should not be severely against the proposed direction.
    if direction == "Haussier":
        if float(last["ema_20"]) <= float(last["ema_50"]) or rsi < 50:
            reasons.append("Le momentum local ne confirme pas suffisamment l'achat.")
            return {**neutral_result, "reasons": reasons, "candle_time": candle_time, "rsi": rsi, "atr": atr, "bias": direction}
        signal = "ACHAT"
        bullish_score, bearish_score = 1, 0
        stop_loss = float(setup["sweep_extreme"]) - 0.2 * atr
        risk_distance = close - stop_loss
        if risk_distance <= 0 or risk_distance > 2.5 * atr:
            reasons.append("Stop structurel trop proche ou trop éloigné : setup rejeté.")
            return {**neutral_result, "reasons": reasons, "candle_time": candle_time, "rsi": rsi, "atr": atr, "bias": direction}
        take_profit = close + 2.0 * risk_distance
    else:
        if float(last["ema_20"]) >= float(last["ema_50"]) or rsi > 50:
            reasons.append("Le momentum local ne confirme pas suffisamment la vente.")
            return {**neutral_result, "reasons": reasons, "candle_time": candle_time, "rsi": rsi, "atr": atr, "bias": direction}
        signal = "VENTE"
        bullish_score, bearish_score = 0, 1
        stop_loss = float(setup["sweep_extreme"]) + 0.2 * atr
        risk_distance = stop_loss - close
        if risk_distance <= 0 or risk_distance > 2.5 * atr:
            reasons.append("Stop structurel trop proche ou trop éloigné : setup rejeté.")
            return {**neutral_result, "reasons": reasons, "candle_time": candle_time, "rsi": rsi, "atr": atr, "bias": direction}
        take_profit = close - 2.0 * risk_distance

    zone_context = assess_smc_zone_confluence(closed, direction, close, atr)
    reasons.append(zone_context["note"])

    reasons.extend([
        f"Biais supérieur aligné : H4={h4_bias}, D1={d1_bias}.",
        "EMA 20/50 et RSI confirment le momentum local.",
        "Stop basé sur l'extrême du sweep avec marge ATR.",
        "Objectif théorique fixé à 2R avant coûts d'exécution.",
    ])
    # This is a transparent rule-based quality score, not a win probability.
    quality = 80
    # MACD contributes only to the transparent setup-quality score.
    if (direction == "Haussier" and last["macd"] > last["macd_signal"]) or (direction == "Baissier" and last["macd"] < last["macd_signal"]):
        quality += 10
    else:
        quality -= 10
    quality += zone_context["score_adjustment"]
    quality = max(0, min(100, quality))
    return {
        "signal": signal,
        "confidence": quality,
        "setup_quality": quality,
        "confirmation": "Confirmée selon les règles ForexScope",
        "bias": direction,
        "entry": close,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_reward": 2.0,
        "smc_zone_context": zone_context,
        "bullish_score": bullish_score,
        "bearish_score": bearish_score,
        "reasons": reasons,
        "rsi": rsi,
        "atr": atr,
        "candle_time": candle_time,
    }


# =========================================================
# 5. PERMANENT SIGNAL HISTORY (SQLite)
# =========================================================
def init_signal_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at_utc TEXT NOT NULL,
                candle_time_utc TEXT,
                market TEXT,
                symbol TEXT,
                timeframe TEXT,
                signal TEXT,
                confidence INTEGER,
                entry REAL,
                stop_loss REAL,
                take_profit REAL,
                reasons TEXT
            )
            """
        )
        conn.commit()


def save_signal_record(market, symbol, timeframe, result, cooldown_minutes=0):
    """
    Enregistre uniquement les signaux ACHAT/VENTE.
    Bloque les répétitions pour le même symbole et la même unité pendant
    le délai configuré. Un même signal issu de la même bougie n'est jamais
    enregistré deux fois.
    """
    signal = result.get("signal")
    created_at = datetime.now(timezone.utc)
    created_at_utc = created_at.isoformat(timespec="seconds")

    if signal not in ("ACHAT", "VENTE"):
        return None, created_at_utc, "not_actionable", 0

    candle_time = result.get("candle_time")
    if candle_time is not None:
        candle_time = pd.to_datetime(candle_time, utc=True, errors="coerce")
        candle_time = candle_time.isoformat() if pd.notna(candle_time) else None

    reasons = "\\n".join(result.get("reasons", []))

    with sqlite3.connect(DB_PATH) as conn:
        # Prevent duplicate records from Streamlit reruns for the same candle.
        if candle_time:
            same_candle = conn.execute(
                """
                SELECT id FROM signals
                WHERE symbol = ? AND timeframe = ? AND candle_time_utc = ?
                  AND signal IN ('ACHAT', 'VENTE')
                ORDER BY id DESC LIMIT 1
                """,
                (symbol, timeframe, candle_time),
            ).fetchone()
            if same_candle:
                return same_candle[0], created_at_utc, "duplicate_candle", 0

        previous = conn.execute(
            """
            SELECT id, created_at_utc, signal
            FROM signals
            WHERE symbol = ? AND timeframe = ?
              AND signal IN ('ACHAT', 'VENTE')
            ORDER BY id DESC LIMIT 1
            """,
            (symbol, timeframe),
        ).fetchone()

        if previous and cooldown_minutes > 0:
            previous_time = pd.to_datetime(previous[1], utc=True, errors="coerce")
            if pd.notna(previous_time):
                elapsed_seconds = max(
                    0.0, (created_at - previous_time.to_pydatetime()).total_seconds()
                )
                remaining_seconds = int(cooldown_minutes * 60 - elapsed_seconds)
                if remaining_seconds > 0:
                    return (
                        previous[0],
                        previous[1],
                        "cooldown",
                        remaining_seconds,
                    )

        cursor = conn.execute(
            """
            INSERT INTO signals (
                created_at_utc, candle_time_utc, market, symbol, timeframe,
                signal, confidence, entry, stop_loss, take_profit, reasons
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at_utc,
                candle_time,
                market,
                symbol,
                timeframe,
                signal,
                result.get("setup_quality", result.get("confidence")),
                result.get("entry"),
                result.get("stop_loss"),
                result.get("take_profit"),
                reasons,
            ),
        )
        conn.commit()
        return cursor.lastrowid, created_at_utc, "saved", 0


def get_signal_history(limit=100):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            """
            SELECT
                id AS ID,
                created_at_utc AS "Créé à (UTC)",
                candle_time_utc AS "Bougie analysée (UTC)",
                market AS Marché,
                symbol AS Symbole,
                timeframe AS Unité,
                signal AS Signal,
                confidence AS "Score enregistré (/100)",
                entry AS Entrée,
                stop_loss AS "Stop Loss",
                take_profit AS "Take Profit",
                reasons AS Raisons
            FROM signals
            ORDER BY id DESC
            LIMIT ?
            """,
            conn,
            params=(int(limit),),
        )


init_signal_db()
# =========================================================
# 6. DISPLAY HELPERS
# =========================================================
def format_price(price):
    if price is None or pd.isna(price):
        return "—"
    price = float(price)
    if abs(price) >= 1000:
        return f"{price:,.2f}"
    if abs(price) >= 10:
        return f"{price:.3f}"
    return f"{price:.5f}"


def detect_fair_value_gaps(data, max_zones=8):
    """Detect simple three-candle FVG zones; these are candidate zones, not guarantees."""
    zones = []
    if data is None or len(data) < 3:
        return zones

    start = max(2, len(data) - 250)
    for i in range(start, len(data)):
        first = data.iloc[i - 2]
        third = data.iloc[i]
        # Bullish gap: current low is above high from two candles earlier.
        if float(third["low"]) > float(first["high"]):
            zones.append({
                "kind": "FVG haussier",
                "low": float(first["high"]),
                "high": float(third["low"]),
                "time": third["time"],
                "color": "rgba(22,163,74,0.18)",
                "line": "#16a34a",
            })
        # Bearish gap: current high is below low from two candles earlier.
        elif float(third["high"]) < float(first["low"]):
            zones.append({
                "kind": "FVG baissier",
                "low": float(third["high"]),
                "high": float(first["low"]),
                "time": third["time"],
                "color": "rgba(220,38,38,0.18)",
                "line": "#dc2626",
            })
    return zones[-max_zones:]


def detect_order_blocks(data, max_zones=6, search_bars=12):
    """Identify candidate OBs as the last opposite candle before a strong break."""
    zones = []
    if data is None or len(data) < 25:
        return zones

    df = calculate_indicators(data)
    start = max(20, len(df) - 180)

    for i in range(start, len(df)):
        row = df.iloc[i]
        atr = row.get("atr")
        if pd.isna(atr) or float(atr) <= 0:
            continue

        body = abs(float(row["close"]) - float(row["open"]))
        bullish_impulse = (
            float(row["close"]) > float(row["open"])
            and body >= 0.8 * float(atr)
        )
        bearish_impulse = (
            float(row["close"]) < float(row["open"])
            and body >= 0.8 * float(atr)
        )

        if not (bullish_impulse or bearish_impulse):
            continue

        left = max(0, i - search_bars)
        candidates = df.iloc[left:i]
        if candidates.empty:
            continue

        if bullish_impulse:
            opposite = candidates[candidates["close"] < candidates["open"]]
            direction = "Order Block haussier"
            color = "rgba(59,130,246,0.16)"
            line = "#3b82f6"
        else:
            opposite = candidates[candidates["close"] > candidates["open"]]
            direction = "Order Block baissier"
            color = "rgba(245,158,11,0.16)"
            line = "#f59e0b"

        if opposite.empty:
            continue

        ob = opposite.iloc[-1]
        zones.append({
            "kind": direction,
            "low": float(ob["low"]),
            "high": float(ob["high"]),
            "time": ob["time"],
            "color": color,
            "line": line,
        })

    # Remove exact duplicate zones, keep the most recent ones.
    unique = {}
    for zone in zones:
        key = (zone["kind"], round(zone["low"], 8), round(zone["high"], 8))
        unique[key] = zone
    return list(unique.values())[-max_zones:]


def build_candlestick_chart(data, market_name, timeframe, signal_result=None,
                            show_fvg=True, show_order_blocks=True,
                            show_liquidity=True):
    fig = go.Figure()
    df = data.copy()

    fig.add_trace(
        go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=market_name,
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
        )
    )

    indicators = calculate_indicators(df)
    for col, label, color in [
        ("ema_20", "EMA 20", "#60a5fa"),
        ("ema_50", "EMA 50", "#fbbf24"),
    ]:
        if col in indicators:
            fig.add_trace(go.Scatter(
                x=indicators["time"], y=indicators[col],
                mode="lines", name=label,
                line=dict(color=color, width=1),
            ))

    if show_fvg:
        for zone in detect_fair_value_gaps(df):
            fig.add_shape(
                type="rect",
                x0=zone["time"], x1=df["time"].iloc[-1],
                y0=zone["low"], y1=zone["high"],
                fillcolor=zone["color"], line=dict(color=zone["line"], width=1),
                layer="below",
            )
            fig.add_annotation(
                x=zone["time"], y=zone["high"], text=zone["kind"],
                showarrow=False, xanchor="left", yanchor="bottom",
                font=dict(size=9, color=zone["line"]),
            )

    if show_order_blocks:
        for zone in detect_order_blocks(df):
            fig.add_shape(
                type="rect",
                x0=zone["time"], x1=df["time"].iloc[-1],
                y0=zone["low"], y1=zone["high"],
                fillcolor=zone["color"], line=dict(color=zone["line"], width=1),
                layer="below",
            )
            fig.add_annotation(
                x=zone["time"], y=zone["low"], text=zone["kind"],
                showarrow=False, xanchor="left", yanchor="top",
                font=dict(size=9, color=zone["line"]),
            )

    if show_liquidity and len(df) >= 20:
        recent = df.tail(20)
        high_level = float(recent["high"].iloc[:-1].max())
        low_level = float(recent["low"].iloc[:-1].min())
        fig.add_hline(
            y=high_level, line_dash="dot", line_color="#a78bfa",
            annotation_text="Liquidité haute", annotation_position="top left",
        )
        fig.add_hline(
            y=low_level, line_dash="dot", line_color="#a78bfa",
            annotation_text="Liquidité basse", annotation_position="bottom left",
        )

    if signal_result and signal_result.get("signal") in ("ACHAT", "VENTE"):
        levels = [
            ("entry", "Entrée", "#38bdf8"),
            ("stop_loss", "Stop Loss", "#ef4444"),
            ("take_profit", "Take Profit", "#22c55e"),
        ]
        for key, label, color in levels:
            value = signal_result.get(key)
            if value is not None and pd.notna(value):
                fig.add_hline(
                    y=float(value), line_dash="dash", line_color=color,
                    annotation_text=f"{label}: {format_price(value)}",
                    annotation_position="right",
                )

    fig.update_layout(
        title=f"{market_name} — {timeframe}",
        xaxis_title="Date et heure (UTC)",
        yaxis_title="Prix",
        template="plotly_dark",
        height=600,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h"),
    )
    return fig


def display_local_time(value):
    if value is None or pd.isna(value):
        return "Indisponible"
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return "Indisponible"
    return parsed.tz_convert("Africa/Kinshasa").strftime("%d/%m/%Y %H:%M:%S")


# =========================================================
# 7. SIDEBAR
# =========================================================
with st.sidebar:
    st.title("ForexScope")
    st.caption("Analyse des marchés financiers")
    st.divider()

    market_name = st.selectbox("Marché", options=list(SYMBOLS.keys()), index=0)
    timeframe = st.selectbox("Unité de temps", options=list(INTERVALS.keys()), index=2)
    cooldown_default = DEFAULT_SIGNAL_COOLDOWN_MINUTES[timeframe]
    cooldown_minutes = st.number_input(
        "Espacement minimal entre signaux (minutes)",
        min_value=0,
        max_value=10080,
        value=int(cooldown_default),
        step=5,
        help="0 désactive le délai. Il s'applique au même symbole et à la même unité.",
    )
    candle_count = st.slider(
        "Nombre de bougies",
        min_value=50,
        max_value=1000,
        value=300,
        step=50,
    )

    if st.button("Actualiser les données", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption(
        "Source : Yahoo Finance. Disponibilité et délais variables "
        "selon le marché."
    )


# =========================================================
# 8. MAIN DASHBOARD
# =========================================================
st.title("ForexScope")
st.caption("Plateforme d'analyse technique et de suivi des marchés.")

tab_market, tab_analysis, tab_history, tab_settings = st.tabs(
    ["📈 Marché", "🎯 Analyse", "📋 Historique", "⚙️ Paramètres"]
)

with st.spinner(f"Chargement des données : {market_name}..."):
    candles = fetch_candles(SYMBOLS[market_name], timeframe, candle_count)

dashboard_checked_at = datetime.now(timezone.utc)

with tab_market:
    st.caption(
        "Dernière vérification de l'application : "
        f"{display_local_time(dashboard_checked_at)} (heure de Kinshasa). "
        "L'heure de la dernière bougie est affichée séparément."
    )
    if candles.empty:
        st.error(
            "Aucune bougie reçue. Vérifie la connexion Internet, "
            "le symbole et l'intervalle choisi."
        )
        st.info(
            "Yahoo Finance limite certaines données intraday. "
            "Cela ne signifie pas nécessairement que le marché est fermé."
        )
    else:
        latest = candles.iloc[-1]
        previous_close = candles.iloc[-2]["close"] if len(candles) >= 2 else latest["close"]
        change = latest["close"] - previous_close
        change_pct = change / previous_close * 100 if previous_close != 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Marché", market_name)
        col2.metric("Dernier prix", format_price(latest["close"]))
        col3.metric(
            "Variation de la dernière bougie",
            format_price(change),
            f"{change_pct:.3f}%",
        )
        col4.metric("Bougies disponibles", f"{len(candles)}")
        st.caption(
            "Heure de la dernière bougie disponible : "
            f"{display_local_time(latest['time'])} (heure de Kinshasa)."
        )

        chart_a, chart_b, chart_c = st.columns(3)
        show_fvg = chart_a.checkbox("Afficher les FVG", value=True, key="show_fvg")
        show_obs = chart_b.checkbox("Afficher les Order Blocks", value=True, key="show_obs")
        show_liquidity = chart_c.checkbox("Afficher les niveaux de liquidité", value=True, key="show_liquidity")

        st.plotly_chart(
            build_candlestick_chart(
                candles, market_name, timeframe,
                show_fvg=show_fvg,
                show_order_blocks=show_obs,
                show_liquidity=show_liquidity,
            ),
            use_container_width=True,
        )

        with st.expander("Consulter les données OHLC"):
            st.dataframe(
                candles.sort_values("time", ascending=False),
                use_container_width=True,
                hide_index=True,
            )


with tab_analysis:
    st.subheader("Analyse technique")
    st.caption(
        "Les signaux exigent un sweep, une cassure de structure et une impulsion, "
        "puis un alignement H4/D1. La qualité du setup n’est pas une probabilité de gain."
    )

    if candles.empty:
        st.warning(
            "Les données du marché sélectionné sont indisponibles. "
            "Impossible de produire une analyse fiable."
        )
    else:
        current_analysis = analyze_timeframe(candles)
        st.markdown("### Synthèse du marché sélectionné")
        st.write(f"**Marché :** {market_name}")
        st.write(f"**Unité de temps :** {timeframe}")

        if current_analysis["status"] != "OK":
            st.warning(current_analysis["status"])
        else:
            a, b, c = st.columns(3)
            a.metric("Biais technique", current_analysis["bias"])
            b.metric("RSI (14)", f"{current_analysis['rsi']:.2f}")
            adx_value = current_analysis["adx"]
            c.metric(
                "ADX (14)",
                f"{adx_value:.2f}" if adx_value is not None else "Indisponible",
            )

            d, e, f = st.columns(3)
            d.metric("Dernière clôture analysée", format_price(current_analysis["close"]))
            e.metric("MACD", current_analysis["macd"])
            f.metric("Structure du marché", current_analysis["structure"])

            st.write("**Balayage de liquidité :** " + current_analysis["sweep"])
            st.write("**ATR (14) :** " + format_price(current_analysis["atr"]))

        st.divider()
        st.markdown("### Analyse multi-unités de temps")

        if st.button(
            "Lancer l'analyse multi-unités",
            key="run_multi_analysis",
            use_container_width=True,
        ):
            with st.spinner("Récupération et analyse des différentes unités..."):
                multi_results = analyze_multiple_timeframes(
                    SYMBOLS[market_name],
                    candle_count,
                )

            rows = []
            for tf, result in multi_results.items():
                bias = result.get("bias", "Neutre")
                if result.get("status") != "OK":
                    verdict = "INDISPONIBLE"
                elif bias == "Haussier":
                    verdict = "SURVEILLER ACHAT"
                elif bias == "Baissier":
                    verdict = "SURVEILLER VENTE"
                else:
                    verdict = "ATTENDRE"
                rows.append({
                    "Unité": tf,
                    "Verdict indicatif": verdict,
                    "État": result.get("status", "Inconnu"),
                    "Biais": bias,
                    "RSI": round(result["rsi"], 2) if result.get("rsi") is not None else None,
                    "ADX": round(result["adx"], 2) if result.get("adx") is not None else None,
                    "MACD": result.get("macd", "Indisponible"),
                    "Structure": result.get("structure", "Indisponible"),
                    "Liquidité": result.get("sweep", "Indisponible"),
                })

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("Le scan couvre M1, M5, M15, M30, H1, H4 et D1. Les unités indisponibles chez Yahoo Finance restent signalées comme telles.")

            valid_results = [
                result for result in multi_results.values()
                if result.get("status") == "OK"
            ]

            if valid_results:
                bullish = sum(r["bias"] == "Haussier" for r in valid_results)
                bearish = sum(r["bias"] == "Baissier" for r in valid_results)
                neutral = sum(r["bias"] == "Neutre" for r in valid_results)

                st.markdown("### Répartition des biais")
                x, y, z = st.columns(3)
                x.metric("Unités haussières", bullish)
                y.metric("Unités baissières", bearish)
                z.metric("Unités neutres", neutral)

                if bullish > bearish:
                    st.info("Les biais haussiers sont majoritaires parmi les unités disponibles.")
                elif bearish > bullish:
                    st.info("Les biais baissiers sont majoritaires parmi les unités disponibles.")
                else:
                    st.info("Les biais haussiers et baissiers sont à égalité parmi les unités disponibles.")
            else:
                st.warning(
                    "Aucune unité de temps n'a fourni assez de données valides pour une analyse."
                )

            st.divider()
            st.markdown("### Signal ForexScope")

            signal_result = generate_trading_signal(candles, multi_results)

            signal_id, created_at_utc, save_status, remaining_seconds = save_signal_record(
                market=market_name,
                symbol=SYMBOLS[market_name],
                timeframe=timeframe,
                result=signal_result,
                cooldown_minutes=int(cooldown_minutes),
            )

            if save_status == "saved":
                st.success(
                    f"Nouveau signal enregistré n°{signal_id} — créé le "
                    f"{display_local_time(created_at_utc)} (heure de Kinshasa)."
                )
            elif save_status == "cooldown":
                remaining_minutes = max(1, int(np.ceil(remaining_seconds / 60)))
                st.info(
                    f"Espacement actif : aucun nouveau signal n'est enregistré "
                    f"sur {market_name} {timeframe} pendant encore environ "
                    f"{remaining_minutes} minute(s). Le signal précédent reste "
                    f"dans l'historique."
                )
            elif save_status == "duplicate_candle":
                st.caption(
                    "Cette bougie a déjà produit un signal enregistré. "
                    "Aucun doublon n'a été ajouté à l'historique."
                )
            elif save_status == "not_actionable":
                st.caption(
                    "Le verdict ATTENDRE n'est pas ajouté à l'historique des "
                    "signaux actionnables."
                )

            s1, s2, s3 = st.columns(3)
            s1.metric("Décision", signal_result["signal"])
            s2.metric("Qualité du setup", f"{signal_result.get('setup_quality', signal_result['confidence'])} / 100")

            rr = signal_result["risk_reward"]
            s3.metric("Ratio risque/rendement", f"1:{rr:.1f}" if rr is not None else "Non défini")

            if signal_result["signal"] == "ACHAT":
                st.success("Biais haussier détecté. Vérifie le contexte avant toute entrée.")
            elif signal_result["signal"] == "VENTE":
                st.error("Biais baissier détecté. Vérifie le contexte avant toute entrée.")
            else:
                st.warning("Pas de confluence suffisante : aucune entrée proposée.")

            if signal_result["signal"] in ("ACHAT", "VENTE"):
                p1, p2, p3 = st.columns(3)
                p1.metric("Entrée indicative", format_price(signal_result["entry"]))
                p2.metric("Stop Loss", format_price(signal_result["stop_loss"]))
                p3.metric("Take Profit", format_price(signal_result["take_profit"]))
            else:
                st.info(
                    "Aucun niveau d'entrée, de SL ou de TP n'est proposé "
                    "tant que le signal reste ATTENDRE."
                )

            with st.expander("Pourquoi ce résultat ?"):
                for reason in signal_result["reasons"]:
                    st.write(f"- {reason}")


with tab_history:
    st.subheader("Historique des signaux")
    history = get_signal_history(limit=100)

    if history.empty:
        st.info("Aucun signal enregistré pour le moment.")
    else:
        history["Créé à"] = history["Créé à (UTC)"].apply(display_local_time)
        history["Bougie analysée"] = history["Bougie analysée (UTC)"].apply(display_local_time)
        history = history.drop(columns=["Créé à (UTC)", "Bougie analysée (UTC)"])

        st.dataframe(history, use_container_width=True, hide_index=True)
        st.download_button(
            label="Télécharger l'historique CSV",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="forexscope_historique.csv",
            mime="text/csv",
        )

with tab_settings:
    st.subheader("Paramètres de données")
    st.write("Fournisseur : Yahoo Finance")
    st.write(f"Symbole Yahoo : {SYMBOLS[market_name]}")
    st.write(f"Unité sélectionnée : {timeframe}")
    st.write(f"Nombre de bougies demandé : {candle_count}")
    st.warning(
        "Les données peuvent être retardées, incomplètes ou indisponibles. "
        "Les bougies H4 sont reconstruites à partir des bougies H1."
    )
    st.caption(
        "Les FVG et Order Blocks affichés sont des zones candidates détectées "
        "par des règles mécaniques. Ils doivent être validés par le contexte "
        "et ne constituent pas, à eux seuls, un signal d'entrée."
    )


st.divider()
st.caption(
    "ForexScope • Page actualisée à : "
    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )


   
