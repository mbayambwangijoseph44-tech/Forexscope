
import json
import os
import sqlite3
import time
from datetime import datetime, timezone

import websocket

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


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

# Symboles Deriv. Certains actifs peuvent ne pas être disponibles sur tous les comptes/régions.
SYMBOLS = {
    "EUR/USD": "frxEURUSD",
    "GBP/USD": "frxGBPUSD",
    "USD/JPY": "frxUSDJPY",
    "AUD/USD": "frxAUDUSD",
    "USD/CAD": "frxUSDCAD",
    "USD/CHF": "frxUSDCHF",
    "NZD/USD": "frxNZDUSD",
    "EUR/GBP": "frxEURGBP",
    "EUR/JPY": "frxEURJPY",
    "GBP/JPY": "frxGBPJPY",
    "Or (Gold)": "frxXAUUSD",
    "Bitcoin / USD": "cryBTCUSD",
}

INTERVALS = {
    "M1": {"granularity": 60},
    "M5": {"granularity": 300},
    "M15": {"granularity": 900},
    "M30": {"granularity": 1800},
    "H1": {"granularity": 3600},
    "H4": {"granularity": 14400},
    "D1": {"granularity": 86400},
}

DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089")
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

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


def _deriv_request(payload, timeout=15):
    """Send one public request to Deriv and always close the socket."""
    ws = None
    try:
        ws = websocket.create_connection(
    DERIV_WS_URL,
    timeout=timeout
)
        ws.send(json.dumps(payload))
        deadline = time.time() + timeout

        while time.time() < deadline:
            response = json.loads(ws.recv())

            if response.get("error"):
                err = response["error"]
                raise RuntimeError(
                    f"Deriv ({err.get('code', 'erreur')}): "
                    f"{err.get('message', 'requête refusée')}"
                )

            if response.get("msg_type"):
                return response

        raise TimeoutError("Délai dépassé en attendant la réponse de Deriv.")

    except Exception as exc:
        raise RuntimeError(str(exc)) from exc

    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass


@st.cache_data(ttl=20, show_spinner=False)
def fetch_candles(symbol, timeframe, count=300):
    """Fetch real OHLC candles from Deriv; never fall back to simulated data."""
    if timeframe not in INTERVALS:
        return pd.DataFrame()

    count = max(50, min(int(count), 1000))

    payload = {
        "ticks_history": symbol,
        "adjust_start_time": 1,
        "count": count,
        "end": "latest",
        "style": "candles",
        "granularity": INTERVALS[timeframe]["granularity"],
        "subscribe": 0,
    }

    try:
        response = _deriv_request(payload)
        candles = response.get("candles", [])

        if not candles:
            return pd.DataFrame()

        data = pd.DataFrame(candles).rename(columns={"epoch": "time"})
        data["time"] = pd.to_datetime(
            data["time"],
            unit="s",
            utc=True,
            errors="coerce",
        )

        return validate_ohlc(data).tail(count).reset_index(drop=True)

    except Exception:
        # Errors are exposed separately by get_connection_status.
        # No other data provider is used.
        return pd.DataFrame()


@st.cache_data(ttl=20, show_spinner=False)
def get_connection_status(symbol):
    """Return explicit Deriv connectivity/data status for the selected symbol."""
    try:
        response = _deriv_request({
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": 2,
            "end": "latest",
            "style": "candles",
            "granularity": 60,
            "subscribe": 0,
        })

        n = len(response.get("candles", []))

        if n:
            return {
                "ok": True,
                "message": "Connexion Deriv active — données de marché reçues.",
            }

        return {
            "ok": False,
            "message": "Deriv a répondu, mais aucune bougie n'est disponible pour ce symbole.",
        }

    except Exception as exc:
        return {
            "ok": False,
            "message": f"Connexion Deriv indisponible : {exc}",
        }


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

    avg_gain = gains.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

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
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()

    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(
        span=9,
        adjust=False,
    ).mean()

    df["macd_hist"] = df["macd"] - df["macd_signal"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0,
        ),
        index=df.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0,
        ),
        index=df.index,
    )

    atr_safe = df["atr"].replace(0, np.nan)

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False,
        ).mean()
        / atr_safe
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False,
        ).mean()
        / atr_safe
    )

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum

    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    df["adx"] = dx.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    return df


def analyze_market_structure(data, lookback=20):
    """Describe a simple close-confirmed break of the prior range."""
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
        result.update(
            structure="Cassure haussière (BOS potentiel)",
            bias="Haussier",
            breakout="HAUSSIER",
        )

    elif close < prior_low:
        result.update(
            structure="Cassure baissière (BOS potentiel)",
            bias="Baissier",
            breakout="BAISSIER",
        )

    else:
        first_close = float(previous["close"].iloc[0])

        if close > first_close:
            result.update(
                structure="Progression haussière sans cassure",
                bias="Haussier",
            )

        elif close < first_close:
            result.update(
                structure="Progression baissière sans cassure",
                bias="Baissier",
            )

        else:
            result.update(
                structure="Range / neutre",
                bias="Neutre",
            )

    return result


def detect_liquidity_sweep(data, lookback=20):
    """Detect a sweep on the final candle of the supplied dataset."""
    result = {
        "sweep": "Aucun balayage détecté",
        "level": None,
        "extreme": None,
        "index": None,
    }

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
        result.update(
            sweep="Sweep haussier potentiel",
            level=previous_low,
            extreme=low,
            index=len(data) - 1,
        )

    elif high > previous_high and close < previous_high:
        result.update(
            sweep="Sweep baissier potentiel",
            level=previous_high,
            extreme=high,
            index=len(data) - 1,
        )

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
    empty = {
        "direction": None,
        "sweep_extreme": None,
        "reasons": [],
    }

    if closed is None or len(closed) < lookback + 3:
        return empty

    # The final candle is the potential confirmation candle.
    # A sweep alone is never an entry signal.
    current = closed.iloc[-1]
    current_index = len(closed) - 1
    first_sweep_index = max(lookback, current_index - sweep_window)

    for sweep_idx in range(
        current_index - 1,
        first_sweep_index - 1,
        -1,
    ):
        sweep_candle = closed.iloc[sweep_idx]
        prior = closed.iloc[max(0, sweep_idx - lookback):sweep_idx]

        if len(prior) < min(lookback, 10):
            continue

        prior_low = float(prior["low"].min())
        prior_high = float(prior["high"].max())

        sweep_low = float(sweep_candle["low"])
        sweep_high = float(sweep_candle["high"])
        sweep_close = float(sweep_candle["close"])

        if sweep_low < prior_low and sweep_close > prior_low:
            direction = "Haussier"
            extreme = sweep_low

            later = closed.iloc[sweep_idx + 1:current_index]

            reference = (
                float(later["high"].max())
                if not later.empty
                else float(prior["high"].tail(5).max())
            )

            broken = float(current["close"]) > reference

        elif sweep_high > prior_high and sweep_close < prior_high:
            direction = "Baissier"
            extreme = sweep_high

            later = closed.iloc[sweep_idx + 1:current_index]

            reference = (
                float(later["low"].min())
                if not later.empty
                else float(prior["low"].tail(5).min())
            )

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
    df = df.dropna(
        subset=["ema_20", "ema_50", "rsi", "atr"],
    )

    if df.empty:
        return {
            **insufficient,
            "status": "Indicateurs indisponibles",
        }

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
        "ema_200": (
            float(last["ema_200"])
            if pd.notna(last["ema_200"])
            else None
        ),
        "bullish_points": bullish_points,
        "bearish_points": bearish_points,
        "candle_time": last["time"],
    }


def analyze_multiple_timeframes(symbol, count=300):
    results = {}

    for tf in ["M5", "M15", "M30", "H1", "H4", "D1"]:
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


# =========================================================
# 4. SIGNAL ENGINE
# =========================================================
def generate_trading_signal(raw_data, multi_results):
    """Generate a conservative signal only after sweep, BOS and impulse align."""
    neutral_result = {
        "signal": "ATTENDRE",
        "confidence": 0,
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
    df = df.dropna(
        subset=["ema_20", "ema_50", "rsi", "atr"],
    )

    # Exclude the newest row because the latest Deriv candle may still be forming.
    closed = df.iloc[:-1].copy()

    if len(closed) < 50:
        return neutral_result

    last = closed.iloc[-1]

    close = float(last["close"])
    atr = float(last["atr"])
    rsi = float(last["rsi"])
    candle_time = last["time"]

    if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
        return {
            **neutral_result,
            "candle_time": candle_time,
        }

    setup = _find_confirmed_smc_setup(closed, atr)
    reasons = list(setup["reasons"])

    if setup["direction"] is None:
        reasons.append(
            "Sweep, cassure de structure et impulsion ne sont pas réunis dans cet ordre."
        )

        return {
            **neutral_result,
            "reasons": reasons,
            "candle_time": candle_time,
            "rsi": rsi,
            "atr": atr,
        }

    direction = setup["direction"]

    # Higher-timeframe context is a filter, not a point added to a loose score.
    h4 = multi_results.get("H4", {}) if isinstance(multi_results, dict) else {}
    d1 = multi_results.get("D1", {}) if isinstance(multi_results, dict) else {}

    h4_bias = h4.get("bias") if h4.get("status") == "OK" else None
    d1_bias = d1.get("bias") if d1.get("status") == "OK" else None

    higher_biases = [
        b for b in (h4_bias, d1_bias)
        if b in ("Haussier", "Baissier")
    ]

    if len(higher_biases) == 2 and higher_biases[0] != higher_biases[1]:
        reasons.append("H4 et D1 sont contradictoires : signal bloqué.")

        return {
            **neutral_result,
            "reasons": reasons,
            "candle_time": candle_time,
            "rsi": rsi,
            "atr": atr,
            "bias": direction,
        }

    if higher_biases and any(b != direction for b in higher_biases):
        reasons.append(
            "Le setup contredit le biais H4/D1 : confirmation insuffisante."
        )

        return {
            **neutral_result,
            "reasons": reasons,
            "candle_time": candle_time,
            "rsi": rsi,
            "atr": atr,
            "bias": direction,
        }

    if not higher_biases:
        reasons.append(
            "Biais H4/D1 indisponible : confirmation supérieure absente."
        )

        return {
            **neutral_result,
            "reasons": reasons,
            "candle_time": candle_time,
            "rsi": rsi,
            "atr": atr,
            "bias": direction,
        }

    # Momentum should not be severely against the proposed direction.
    if direction == "Haussier":
        if float(last["ema_20"]) <= float(last["ema_50"]) or rsi < 50:
            reasons.append(
                "Le momentum local ne confirme pas suffisamment l'achat."
            )

            return {
                **neutral_result,
                "reasons": reasons,
                "candle_time": candle_time,
                "rsi": rsi,
                "atr": atr,
                "bias": direction,
            }

        signal = "ACHAT"
        bullish_score, bearish_score = 1, 0

        stop_loss = float(setup["sweep_extreme"]) - 0.2 * atr
        risk_distance = close - stop_loss

        if risk_distance <= 0 or risk_distance > 2.5 * atr:
            reasons.append(
                "Stop structurel trop proche ou trop éloigné : setup rejeté."
            )

            return {
                **neutral_result,
                "reasons": reasons,
                "candle_time": candle_time,
                "rsi": rsi,
                "atr": atr,
                "bias": direction,
            }

        take_profit = close + 2.0 * risk_distance

    else:
        if float(last["ema_20"]) >= float(last["ema_50"]) or rsi > 50:
            reasons.append(
                "Le momentum local ne confirme pas suffisamment la vente."
            )

            return {
                **neutral_result,
                "reasons": reasons,
                "candle_time": candle_time,
                "rsi": rsi,
                "atr": atr,
                "bias": direction,
            }

        signal = "VENTE"
        bullish_score, bearish_score = 0, 1

        stop_loss = float(setup["sweep_extreme"]) + 0.2 * atr
        risk_distance = stop_loss - close

        if risk_distance <= 0 or risk_distance > 2.5 * atr:
            reasons.append(
                "Stop structurel trop proche ou trop éloigné : setup rejeté."
            )

            return {
                **neutral_result,
                "reasons": reasons,
                "candle_time": candle_time,
                "rsi": rsi,
                "atr": atr,
                "bias": direction,
            }

        take_profit = close - 2.0 * risk_distance

    reasons.extend([
        f"Biais supérieur aligné : H4={h4_bias}, D1={d1_bias}.",
        "EMA 20/50 et RSI confirment le momentum local.",
        "Stop basé sur l'extrême du sweep avec marge ATR.",
        "Objectif théorique fixé à 2R avant coûts d'exécution.",
    ])

    # This is a rule-based quality score, not a win probability.
    quality = 80

    if (
        direction == "Haussier"
        and last["macd"] > last["macd_signal"]
    ) or (
        direction == "Baissier"
        and last["macd"] < last["macd_signal"]
    ):
        quality += 10
    else:
        quality -= 10

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


def save_signal_record(market, symbol, timeframe, result):
    created_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    candle_time = result.get("candle_time")

    if candle_time is not None:
        candle_time = pd.to_datetime(
            candle_time,
            utc=True,
            errors="coerce",
        )
        candle_time = (
            candle_time.isoformat()
            if pd.notna(candle_time)
            else None
        )

    reasons = "\n".join(result.get("reasons", []))

    with sqlite3.connect(DB_PATH) as conn:
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
                result.get("signal"),
                result.get("setup_quality", result.get("confidence")),
                result.get("entry"),
                result.get("stop_loss"),
                result.get("take_profit"),
                reasons,
            ),
        )

        conn.commit()
        return cursor.lastrowid, created_at_utc


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


def build_candlestick_chart(data, market_name, timeframe):
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=data["time"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name=market_name,
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
        )
    )

    fig.update_layout(
        title=f"{market_name} — {timeframe}",
        xaxis_title="Date et heure (UTC)",
        yaxis_title="Prix",
        template="plotly_dark",
        height=550,
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

    return parsed.tz_convert("Africa/Kinshasa").strftime(
        "%d/%m/%Y %H:%M:%S"
    )


# =========================================================
# 7. SIDEBAR
# =========================================================
with st.sidebar:
    st.title("ForexScope")
    st.caption("Analyse des marchés financiers")
    st.divider()

    market_name = st.selectbox(
        "Marché",
        options=list(SYMBOLS.keys()),
        index=0,
    )

    timeframe = st.selectbox(
        "Unité de temps",
        options=list(INTERVALS.keys()),
        index=2,
    )

    candle_count = st.slider(
        "Nombre de bougies",
        min_value=50,
        max_value=1000,
        value=300,
        step=50,
    )

    if st.button(
        "Actualiser les données",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    st.caption(
        "Source unique : Deriv WebSocket. Aucune source de secours."
    )


# =========================================================
# 8. MAIN DASHBOARD
# =========================================================
st.title("ForexScope")
st.caption(
    "Plateforme d'analyse technique et de suivi des marchés."
)

tab_market, tab_analysis, tab_history, tab_settings = st.tabs(
    ["📈 Marché", "🎯 Analyse", "📋 Historique", "⚙️ Paramètres"]
)

with st.spinner(f"Chargement des données Deriv : {market_name}..."):
    candles = fetch_candles(
        SYMBOLS[market_name],
        timeframe,
        candle_count,
    )

    connection_status = get_connection_status(
        SYMBOLS[market_name]
    )

if connection_status["ok"]:
    st.success(connection_status["message"])
else:
    st.error(connection_status["message"])


with tab_market:
    if candles.empty:
        st.error(
            "Aucune bougie reçue. Vérifie la connexion Internet, "
            "le symbole et l'intervalle choisi."
        )

        st.info(
            "Aucune donnée Deriv reçue. Vérifie le symbole, "
            "l’intervalle et la connexion."
        )

    else:
        latest = candles.iloc[-1]

        previous_close = (
            candles.iloc[-2]["close"]
            if len(candles) >= 2
            else latest["close"]
        )

        change = latest["close"] - previous_close

        change_pct = (
            change / previous_close * 100
            if previous_close != 0
            else 0
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Marché", market_name)
        col2.metric("Dernier prix", format_price(latest["close"]))

        col3.metric(
            "Variation de la dernière bougie",
            format_price(change),
            f"{change_pct:.3f}%",
        )

        col4.metric(
            "Bougies disponibles",
            f"{len(candles)}",
        )

        st.plotly_chart(
            build_candlestick_chart(
                candles,
                market_name,
                timeframe,
            ),
            use_container_width=True,
        )

        with st.expander("Consulter les données OHLC"):
            st.dataframe(
                candles.sort_values(
                    "time",
                    ascending=False,
                ),
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

            a.metric(
                "Biais technique",
                current_analysis["bias"],
            )

            b.metric(
                "RSI (14)",
                f"{current_analysis['rsi']:.2f}",
            )

            adx_value = current_analysis["adx"]

            c.metric(
                "ADX (14)",
                f"{adx_value:.2f}"
                if adx_value is not None
                else "Indisponible",
            )

            d, e, f = st.columns(3)

            d.metric(
                "Dernière clôture analysée",
                format_price(current_analysis["close"]),
            )

            e.metric(
                "MACD",
                current_analysis["macd"],
            )

            f.metric(
                "Structure du marché",
                current_analysis["structure"],
            )

            st.write(
                "**Balayage de liquidité :** "
                + current_analysis["sweep"]
            )

            st.write(
                "**ATR (14) :** "
                + format_price(current_analysis["atr"])
            )

        st.divider()
        st.markdown("### Analyse multi-unités de temps")

        if st.button(
            "Lancer l'analyse multi-unités",
            key="run_multi_analysis",
            use_container_width=True,
        ):
            with st.spinner(
                "Récupération et analyse des différentes unités..."
            ):
                multi_results = analyze_multiple_timeframes(
                    SYMBOLS[market_name],
                    candle_count,
                )

            rows = []

            for tf, result in multi_results.items():
                rows.append({
                    "Unité": tf,
                    "État": result.get("status", "Inconnu"),
                    "Biais": result.get("bias", "Neutre"),
                    "RSI": (
                        round(result["rsi"], 2)
                        if result.get("rsi") is not None
                        else None
                    ),
                    "ADX": (
                        round(result["adx"], 2)
                        if result.get("adx") is not None
                        else None
                    ),
                    "MACD": result.get("macd", "Indisponible"),
                    "Structure": result.get(
                        "structure",
                        "Indisponible",
                    ),
                    "Liquidité": result.get(
                        "sweep",
                        "Indisponible",
                    ),
                })

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

            valid_results = [
                result
                for result in multi_results.values()
                if result.get("status") == "OK"
            ]

            if valid_results:
                bullish = sum(
                    r["bias"] == "Haussier"
                    for r in valid_results
                )

                bearish = sum(
                    r["bias"] == "Baissier"
                    for r in valid_results
                )

                neutral = sum(
                    r["bias"] == "Neutre"
                    for r in valid_results
                )

                st.markdown("### Répartition des biais")

                x, y, z = st.columns(3)

                x.metric("Unités haussières", bullish)
                y.metric("Unités baissières", bearish)
                z.metric("Unités neutres", neutral)

                if bullish > bearish:
                    st.info(
                        "Les biais haussiers sont majoritaires "
                        "parmi les unités disponibles."
                    )

                elif bearish > bullish:
                    st.info(
                        "Les biais baissiers sont majoritaires "
                        "parmi les unités disponibles."
                    )

                else:
                    st.info(
                        "Les biais haussiers et baissiers sont "
                        "à égalité parmi les unités disponibles."
                    )

            else:
                st.warning(
                    "Aucune unité de temps n'a fourni assez de "
                    "données valides pour une analyse."
                )

            st.divider()
            st.markdown("### Signal ForexScope")

            signal_result = generate_trading_signal(
                candles,
                multi_results,
            )

            signal_id, created_at_utc = save_signal_record(
                market=market_name,
                symbol=SYMBOLS[market_name],
                timeframe=timeframe,
                result=signal_result,
            )

            st.caption(
                f"Signal enregistré n°{signal_id} — créé le "
                f"{display_local_time(created_at_utc)} "
                "(heure de Kinshasa)"
            )

            s1, s2, s3 = st.columns(3)

            s1.metric(
                "Décision",
                signal_result["signal"],
            )

            s2.metric(
                "Qualité du setup",
                f"{signal_result.get('setup_quality', signal_result['confidence'])} / 100",
            )

            rr = signal_result["risk_reward"]

            s3.metric(
                "Ratio risque/rendement",
                f"1:{rr:.1f}" if rr is not None else "Non défini",
            )

            if signal_result["signal"] == "ACHAT":
                st.success(
                    "Biais haussier détecté. Vérifie le contexte avant toute entrée."
                )

            elif signal_result["signal"] == "VENTE":
                st.error(
                    "Biais baissier détecté. Vérifie le contexte avant toute entrée."
                )

            else:
                st.warning(
                    "Pas de confluence suffisante : aucune entrée proposée."
                )

            if signal_result["signal"] in ("ACHAT", "VENTE"):
                p1, p2, p3 = st.columns(3)

                p1.metric(
                    "Entrée indicative",
                    format_price(signal_result["entry"]),
                )

                p2.metric(
                    "Stop Loss",
                    format_price(signal_result["stop_loss"]),
                )

                p3.metric(
                    "Take Profit",
                    format_price(signal_result["take_profit"]),
                )

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
        history["Créé à"] = history["Créé à (UTC)"].apply(
            display_local_time
        )

        history["Bougie analysée"] = history[
            "Bougie analysée (UTC)"
        ].apply(display_local_time)

        history = history.drop(
            columns=[
                "Créé à (UTC)",
                "Bougie analysée (UTC)",
            ]
        )

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            label="Télécharger l'historique CSV",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="forexscope_historique.csv",
            mime="text/csv",
        )


with tab_settings:
    st.subheader("Paramètres de données")

    st.write("Fournisseur : Deriv WebSocket (données publiques)")
    st.write(f"Symbole Deriv : {SYMBOLS[market_name]}")
    st.write(f"Unité sélectionnée : {timeframe}")
    st.write(f"Nombre de bougies demandé : {candle_count}")

    st.warning(
        "La disponibilité dépend du symbole Deriv et des autorisations du service. "
        "ForexScope n’utilise aucune donnée simulée ni aucun repli vers Yahoo Finance."
    )


st.divider()

st.caption(
    "ForexScope • Page actualisée à : "
    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
    
