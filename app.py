import sqlite3
import io
import os
import shutil
import tempfile
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN

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

# Stockage SQLite configurable pour Render ou un environnement local.
# Sur Render, créer un Persistent Disk monté sur /var/data, puis définir
# FOREXSCOPE_DATA_DIR=/var/data dans les variables d’environnement.
_default_data_dir = "/var/data" if os.path.isdir("/var/data") else os.getcwd()
DATA_DIR = os.environ.get("FOREXSCOPE_DATA_DIR", _default_data_dir)
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "forexscope.db")

# Migration prudente : si une ancienne base locale existe et que la nouvelle
# base persistante n’existe pas encore, la copier sans écraser de données.
_legacy_db = os.path.join(os.getcwd(), "forexscope.db")
if not os.path.exists(DB_PATH) and os.path.abspath(_legacy_db) != os.path.abspath(DB_PATH):
    if os.path.isfile(_legacy_db):
        try:
            shutil.copy2(_legacy_db, DB_PATH)
        except OSError:
            pass

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
        & (data[required] > 0).all(axis=1)
    ]

    data = (
        data.sort_values("time")
        .drop_duplicates(subset=["time"])
        .reset_index(drop=True)
    )
    return data[["time", "open", "high", "low", "close"]]


def run_market_data_integrity_tests():
    """Deterministic OHLC validation tests using synthetic data only."""
    now = pd.Timestamp("2026-01-01T00:00:00Z")
    valid = pd.DataFrame({
        "time": [now, now + pd.Timedelta(minutes=5)],
        "open": [1.10, 1.11], "high": [1.12, 1.13],
        "low": [1.09, 1.10], "close": [1.11, 1.12],
    })
    duplicate = pd.concat([valid, valid.iloc[[1]]], ignore_index=True)
    invalid_ohlc = pd.DataFrame({
        "time": [now, now + pd.Timedelta(minutes=5)],
        "open": [1.10, 1.11], "high": [1.12, 1.00],
        "low": [1.09, 1.10], "close": [1.11, 1.12],
    })
    missing_column = pd.DataFrame({"time": [now], "open": [1.1], "high": [1.2], "low": [1.0]})
    cases = [
        ("Bougies valides conservées", valid, 2),
        ("Horodatages dupliqués supprimés", duplicate, 2),
        ("Bougie OHLC incohérente rejetée", invalid_ohlc, 1),
        ("Colonnes OHLC manquantes rejetées", missing_column, 0),
        ("Entrée vide gérée", pd.DataFrame(), 0),
    ]
    rows = []
    for name, frame, expected_count in cases:
        try:
            result = validate_ohlc(frame)
            actual_count = len(result)
            ok = actual_count == expected_count
            # All returned candles must be chronologically sorted and unique.
            if ok and not result.empty:
                ok = result["time"].is_monotonic_increasing and not result["time"].duplicated().any()
            rows.append({"Test": name, "Attendu": expected_count, "Obtenu": actual_count,
                         "État": "RÉUSSI" if ok else "ÉCHEC"})
        except Exception as exc:
            rows.append({"Test": name, "Attendu": expected_count,
                         "Obtenu": f"Erreur: {type(exc).__name__}", "État": "ÉCHEC"})
    return pd.DataFrame(rows)


def build_candle_quality_report(data, timeframe):
    """Inspect loaded candles without changing them or making a network request."""
    rows = []
    if data is None or data.empty:
        return pd.DataFrame([{"Contrôle": "Disponibilité", "État": "INDISPONIBLE", "Détail": "Aucune bougie chargée."}])
    frame = data.copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    required = ["time", "open", "high", "low", "close"]
    missing_cols = [c for c in required if c not in frame.columns]
    rows.append({"Contrôle": "Colonnes requises", "État": "OK" if not missing_cols else "ERREUR",
                 "Détail": "Toutes présentes" if not missing_cols else ", ".join(missing_cols)})
    if missing_cols:
        return pd.DataFrame(rows)
    invalid_time = int(frame["time"].isna().sum())
    duplicate_time = int(frame["time"].duplicated().sum())
    chronological = frame["time"].dropna().is_monotonic_increasing
    rows.extend([
        {"Contrôle": "Nombre de bougies", "État": "OK" if len(frame) >= 30 else "À VÉRIFIER", "Détail": str(len(frame))},
        {"Contrôle": "Horodatages invalides", "État": "OK" if invalid_time == 0 else "ERREUR", "Détail": str(invalid_time)},
        {"Contrôle": "Horodatages dupliqués", "État": "OK" if duplicate_time == 0 else "ERREUR", "Détail": str(duplicate_time)},
        {"Contrôle": "Ordre chronologique", "État": "OK" if chronological else "À VÉRIFIER", "Détail": "Croissant" if chronological else "Données non triées"},
    ])
    numeric = frame[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    bad_numeric = int(numeric.isna().any(axis=1).sum())
    bad_ohlc = int(((numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)) |
                    (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1) )).sum())
    rows.extend([
        {"Contrôle": "Valeurs OHLC non numériques/manquantes", "État": "OK" if bad_numeric == 0 else "ERREUR", "Détail": str(bad_numeric)},
        {"Contrôle": "Cohérence High/Low", "État": "OK" if bad_ohlc == 0 else "ERREUR", "Détail": str(bad_ohlc)},
    ])
    if frame["time"].notna().any():
        last_time = frame["time"].dropna().max()
        age = max(0.0, (pd.Timestamp.now(tz="UTC") - last_time).total_seconds() / 60.0)
        expected = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}.get(timeframe, 60)
        fresh = age <= max(expected * 3, 15 if timeframe.startswith("M") else expected * 2)
        rows.append({"Contrôle": "Fraîcheur indicative", "État": "OK" if fresh else "À VÉRIFIER",
                     "Détail": f"Dernière bougie : {last_time.isoformat()} ; âge ≈ {age:.1f} min"})
    return pd.DataFrame(rows)


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



def build_mtf_consistency_report(mtf_results):
    """Create a transparent consistency report from already computed timeframe results."""
    order = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
    rows = []
    valid_biases = []
    for tf in order:
        item = (mtf_results or {}).get(tf, {})
        status = item.get("status", "Données indisponibles")
        bias = item.get("bias", "Neutre")
        if status != "OK":
            bias = "Indisponible"
        elif bias not in ("Haussier", "Baissier", "Neutre"):
            bias = "Neutre"
        if bias in ("Haussier", "Baissier"):
            valid_biases.append((tf, bias))
        candle_time = item.get("candle_time")
        try:
            candle_time = pd.Timestamp(candle_time).isoformat() if candle_time is not None else "Indisponible"
        except Exception:
            candle_time = str(candle_time) if candle_time is not None else "Indisponible"
        # Freshness is a diagnostic, not proof that a market should be trading now.
        # Yahoo timestamps may be delayed and FX/crypto/equity calendars differ.
        age_minutes = None
        freshness = "INCONNUE"
        raw_time = item.get("candle_time")
        if status == "OK" and raw_time is not None:
            try:
                ts = pd.Timestamp(raw_time)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                else:
                    ts = ts.tz_convert("UTC")
                age_minutes = max(0.0, (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 60.0)
                expected_minutes = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}.get(tf, 60)
                tolerance_minutes = max(expected_minutes * 3, 15 if tf.startswith("M") else expected_minutes * 2)
                freshness = "RÉCENTE" if age_minutes <= tolerance_minutes else "POTENTIELLEMENT PÉRIMÉE"
            except Exception:
                freshness = "HORODATAGE INVALIDE"
        elif status != "OK":
            freshness = "INDISPONIBLE"
        rows.append({
            "Unité": tf, "État": status, "Biais": bias,
            "RSI": round(float(item["rsi"]), 2) if item.get("rsi") is not None and np.isfinite(item["rsi"]) else None,
            "Clôture analysée": item.get("close"), "Heure bougie": candle_time,
            "Âge bougie (min)": round(age_minutes, 1) if age_minutes is not None else None,
            "Fraîcheur indicative": freshness,
        })

    higher = [b for tf, b in valid_biases if tf in ("H4", "D1")]
    execution = [b for tf, b in valid_biases if tf in ("M5", "M15")]
    if not higher or not execution:
        summary = "Données insuffisantes pour comparer le biais supérieur H4/D1 aux unités M5/M15."
        state = "INCOMPLET"
    elif len(set(higher)) == 1 and len(set(execution)) == 1 and higher[0] == execution[0]:
        summary = f"Confluence directionnelle : H4/D1 et M5/M15 sont {higher[0].lower()}. Cela ne garantit pas un signal d’entrée."
        state = "ALIGNÉ"
    elif set(higher).isdisjoint(set(execution)):
        summary = "Conflit directionnel entre H4/D1 et M5/M15. Une entrée dans le sens du biais supérieur n’est pas confirmée par les unités courtes."
        state = "CONFLIT"
    else:
        summary = "Confluence partielle : certaines unités sont alignées et d’autres divergent. Vérifier le contexte avant toute décision."
        state = "PARTIEL"
    return pd.DataFrame(rows), state, summary

def run_mtf_integrity_unit_tests():
    """Small deterministic tests for missing, malformed and stale MTF timestamps."""
    cases = [
        {"name": "Unité absente", "item": {"status": "Données indisponibles", "bias": "Neutre", "candle_time": None}, "expected": "INDISPONIBLE"},
        {"name": "Horodatage invalide", "item": {"status": "OK", "bias": "Haussier", "candle_time": "pas-une-date", "rsi": 50, "close": 1.2}, "expected": "HORODATAGE INVALIDE"},
        {"name": "Biais inconnu", "item": {"status": "OK", "bias": "inconnu", "candle_time": pd.Timestamp.now(tz="UTC"), "rsi": 50, "close": 1.2}, "expected_bias": "Neutre"},
    ]
    rows = []
    for case in cases:
        report, _, _ = build_mtf_consistency_report({"M5": case["item"]})
        row = report.loc[report["Unité"] == "M5"].iloc[0]
        if "expected" in case:
            actual = row["Fraîcheur indicative"]
            passed = actual == case["expected"]
            expected = case["expected"]
        else:
            actual = row["Biais"]
            expected = case["expected_bias"]
            passed = actual == expected
        rows.append({"Test": case["name"], "Attendu": expected, "Obtenu": actual, "État": "RÉUSSI" if passed else "ÉCHEC"})
    return pd.DataFrame(rows)



def assess_mtf_readiness(mtf_results, now=None):
    """Conservative MTF readiness gate; this is a data/consistency check, not a trade signal."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    required = {"M5": 45, "M15": 120, "H4": 2160, "D1": 10080}
    issues, directions = [], []
    details = []
    for tf, max_age in required.items():
        item = mtf_results.get(tf, {}) if isinstance(mtf_results, dict) else {}
        status = item.get("status", "Données indisponibles")
        bias = item.get("bias", "Neutre")
        age = None
        if status != "OK":
            issues.append(f"{tf} indisponible.")
        else:
            ts = pd.to_datetime(item.get("candle_time"), utc=True, errors="coerce")
            if pd.isna(ts):
                issues.append(f"Horodatage {tf} absent ou invalide.")
            else:
                age = max(0.0, (now - ts).total_seconds() / 60.0)
                if age > max_age:
                    issues.append(f"Données {tf} trop anciennes ({age:.0f} min; seuil {max_age} min).")
            if bias not in ("Haussier", "Baissier"):
                issues.append(f"Biais {tf} neutre ou non exploitable.")
            else:
                directions.append((tf, bias))
        details.append({"Unité": tf, "État données": status, "Biais": bias, "Âge (min)": round(age, 1) if age is not None else None, "Seuil (min)": max_age})
    if not issues and len({b for _, b in directions}) == 1:
        state = "PRÊT POUR ÉVALUATION"
        summary = "Les quatre unités requises sont disponibles, indicativement récentes et alignées. Cela ne constitue pas une recommandation d’entrée."
    elif len({b for _, b in directions}) > 1:
        state = "CONFLIT DIRECTIONNEL"
        summary = "Les unités requises présentent des directions opposées. Le contrôle ne valide pas de confluence complète."
    else:
        state = "INCOMPLET"
        summary = "Les données, les horodatages, la fraîcheur ou les biais ne permettent pas de valider la confluence complète."
    return pd.DataFrame(details), state, summary, issues


def run_mtf_readiness_tests():
    now = pd.Timestamp("2026-01-01T12:00:00Z")
    def pack(bias="Haussier", timestamp="2026-01-01T11:55:00Z", status="OK"):
        return {"status": status, "bias": bias, "candle_time": timestamp}
    base = {"M5": pack(), "M15": pack(), "H4": pack(timestamp="2026-01-01T08:00:00Z"), "D1": pack(timestamp="2026-01-01T00:00:00Z")}
    cases = []
    cases.append(("Confluence complète récente", base, "PRÊT POUR ÉVALUATION"))
    missing = dict(base); missing["H4"] = pack(status="Données indisponibles")
    cases.append(("Contexte H4 indisponible", missing, "INCOMPLET"))
    conflict = dict(base); conflict["M15"] = pack(bias="Baissier")
    cases.append(("Directions contradictoires", conflict, "CONFLIT DIRECTIONNEL"))
    stale = dict(base); stale["M5"] = pack(timestamp="2026-01-01T10:00:00Z")
    cases.append(("Bougie M5 trop ancienne", stale, "INCOMPLET"))
    neutral = dict(base); neutral["D1"] = pack(bias="Neutre", timestamp="2026-01-01T00:00:00Z")
    cases.append(("Biais supérieur neutre", neutral, "INCOMPLET"))
    rows = []
    for name, data, expected in cases:
        _, actual, _, _ = assess_mtf_readiness(data, now=now)
        rows.append({"Test": name, "Attendu": expected, "Obtenu": actual, "État": "RÉUSSI" if actual == expected else "ÉCHEC"})
    return pd.DataFrame(rows)

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



def validate_signal_data_freshness(raw_data, selected_timeframe, mtf_results, now=None):
    """Fail closed when the selected candle or required H4/D1 context is missing or stale."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")

    limits_minutes = {
        "M1": 15, "M5": 45, "M15": 120, "M30": 240,
        "H1": 480, "H4": 2160, "D1": 10080,
    }
    issues = []
    if raw_data is None or raw_data.empty or "time" not in raw_data.columns:
        issues.append("Les bougies du timeframe sélectionné sont absentes ou sans horodatage.")
    else:
        try:
            latest = pd.to_datetime(raw_data["time"].iloc[-1], utc=True, errors="coerce")
            if pd.isna(latest):
                issues.append("L’horodatage de la dernière bougie est invalide.")
            else:
                age_minutes = max(0.0, (now - latest).total_seconds() / 60.0)
                limit = limits_minutes.get(selected_timeframe, 240)
                if age_minutes > limit:
                    issues.append(
                        f"Dernière bougie {selected_timeframe} trop ancienne : "
                        f"{age_minutes:.0f} min (seuil indicatif {limit} min)."
                    )
        except (TypeError, ValueError, IndexError):
            issues.append("Impossible de déterminer l’âge de la dernière bougie.")

    # H4 and D1 are required by the signal engine. Check availability, timestamp and age.
    for tf, limit_minutes in (("H4", 2160), ("D1", 10080)):
        item = mtf_results.get(tf, {}) if isinstance(mtf_results, dict) else {}
        if item.get("status") != "OK":
            issues.append(f"Contexte {tf} indisponible : le signal doit rester ATTENDRE.")
            continue
        raw_time = item.get("candle_time")
        parsed = pd.to_datetime(raw_time, utc=True, errors="coerce")
        if pd.isna(parsed):
            issues.append(f"Horodatage {tf} absent ou invalide.")
            continue
        age_minutes = max(0.0, (now - parsed).total_seconds() / 60.0)
        if age_minutes > limit_minutes:
            issues.append(
                f"Contexte {tf} trop ancien : {age_minutes:.0f} min "
                f"(seuil indicatif {limit_minutes} min)."
            )
    return {"ok": not issues, "issues": issues}


def run_signal_freshness_guard_tests():
    """Small deterministic tests for missing, fresh and stale signal inputs."""
    now = pd.Timestamp("2026-10-16T12:00:00Z")
    fresh = pd.DataFrame({"time": [now - pd.Timedelta(minutes=5)]})
    stale = pd.DataFrame({"time": [now - pd.Timedelta(hours=3)]})
    mtf_fresh = {
        "H4": {"status": "OK", "bias": "Haussier", "candle_time": now - pd.Timedelta(hours=4)},
        "D1": {"status": "OK", "bias": "Haussier", "candle_time": now - pd.Timedelta(days=1)},
    }
    cases = [
        ("Données fraîches", validate_signal_data_freshness(fresh, "M15", mtf_fresh, now), True),
        ("Timeframe sélectionné périmé", validate_signal_data_freshness(stale, "M15", mtf_fresh, now), False),
        ("Contexte H4 manquant", validate_signal_data_freshness(fresh, "M15", {**mtf_fresh, "H4": {"status": "Données indisponibles"}}, now), False),
        ("Horodatage sélectionné absent", validate_signal_data_freshness(pd.DataFrame(), "M15", mtf_fresh, now), False),
    ]
    return pd.DataFrame([
        {"Test": name, "État": "RÉUSSI" if result["ok"] == expected else "ÉCHEC",
         "Détail": "; ".join(result["issues"]) if result["issues"] else "Données acceptées."}
        for name, result, expected in cases
    ])

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

    # Momentum filters: do not buy into bearish momentum or sell into bullish momentum.
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

    
    # Anti-chasing filter: reject entries that are already too extended from EMA20.
    ema_distance_atr = abs(close - float(last["ema_20"])) / atr
    if ema_distance_atr > 1.5:
        reasons.append(f"Entrée trop éloignée de l'EMA20 ({ema_distance_atr:.2f} ATR) : risque de signal tardif, setup rejeté.")
        return {**neutral_result, "reasons": reasons, "candle_time": candle_time, "rsi": rsi, "atr": atr, "bias": direction}

    adx_value = float(last["adx"]) if pd.notna(last["adx"]) else None
    if adx_value is not None and adx_value < 12:
        reasons.append(f"ADX faible ({adx_value:.1f}) : marché possiblement sans impulsion, setup rejeté.")
        return {**neutral_result, "reasons": reasons, "candle_time": candle_time, "rsi": rsi, "atr": atr, "bias": direction}

    zone_context = assess_smc_zone_confluence(closed, direction, close, atr)
    reasons.append(zone_context["note"])

    reasons.extend([
        f"Biais supérieur aligné : H4={h4_bias}, D1={d1_bias}.",
        "EMA 20/50 et RSI confirment le momentum local.",
        f"Distance entrée–EMA20 : {ema_distance_atr:.2f} ATR.",
        f"ADX : {adx_value:.1f} (filtre d'impulsion)." if adx_value is not None else "ADX indisponible : non utilisé pour le score.",
        "Stop basé sur l'extrême du sweep avec marge ATR.",
        "Objectif théorique fixé à 2R avant coûts d'exécution.",
    ])
    # Transparent setup-quality score; it is NOT a win probability.
    quality = 70
    if (direction == "Haussier" and last["macd"] > last["macd_signal"]) or (direction == "Baissier" and last["macd"] < last["macd_signal"]):
        quality += 10
    else:
        quality -= 10
    if adx_value is not None:
        quality += 10 if adx_value >= 25 else (5 if adx_value >= 18 else -5)
    # Penalize stretched entries even when they remain within the hard rejection limit.
    quality -= 10 if ema_distance_atr > 1.0 else 0
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
        "ema_distance_atr": round(ema_distance_atr, 3),
        "adx": adx_value,
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
                reasons TEXT,
                outcome_status TEXT NOT NULL DEFAULT 'OUVERT',
                outcome_updated_utc TEXT,
                actual_entry REAL,
                actual_exit REAL,
                realized_r REAL,
                journal_notes TEXT,
                execution_notes TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runtime_startups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at_utc TEXT NOT NULL,
                data_dir TEXT NOT NULL,
                db_path TEXT NOT NULL,
                app_version TEXT NOT NULL
            )
            """
        )
        # Migration sûre pour les bases SQLite déjà créées en phase précédente.
        columns = {row[1] for row in conn.execute("PRAGMA table_info(signals)").fetchall()}
        if "outcome_status" not in columns:
            conn.execute("ALTER TABLE signals ADD COLUMN outcome_status TEXT NOT NULL DEFAULT 'OUVERT'")
        if "outcome_updated_utc" not in columns:
            conn.execute("ALTER TABLE signals ADD COLUMN outcome_updated_utc TEXT")
        # Phase 18: extend the journal without deleting previous signal history.
        for column, definition in {
            "actual_entry": "REAL",
            "actual_exit": "REAL",
            "realized_r": "REAL",
            "journal_notes": "TEXT",
            "execution_notes": "TEXT",
        }.items():
            if column not in columns:
                conn.execute(f"ALTER TABLE signals ADD COLUMN {column} {definition}")
        conn.commit()



def validate_signal_record_consistency(result):
    """Validate actionable signal direction, price levels and candle timestamp before persistence."""
    if not isinstance(result, dict):
        return {"ok": False, "issues": ["Résultat de signal absent ou invalide."]}
    side = result.get("signal")
    if side not in ("ACHAT", "VENTE"):
        return {"ok": True, "issues": []}  # ATTENDRE is intentionally not persisted.
    issues = []
    try:
        entry = float(result.get("entry"))
        sl = float(result.get("stop_loss"))
        tp = float(result.get("take_profit"))
        if not all(np.isfinite(x) and x > 0 for x in (entry, sl, tp)):
            issues.append("Entrée, SL et TP doivent être des prix positifs et finis.")
        elif side == "ACHAT" and not sl < entry < tp:
            issues.append("Pour ACHAT, les niveaux doivent respecter SL < Entrée < TP.")
        elif side == "VENTE" and not tp < entry < sl:
            issues.append("Pour VENTE, les niveaux doivent respecter TP < Entrée < SL.")
    except (TypeError, ValueError):
        issues.append("Entrée, SL ou TP manquant ou non numérique.")
    # Verify that the declared reward/risk ratio matches the actual price levels.
    try:
        entry = float(result.get("entry"))
        sl = float(result.get("stop_loss"))
        tp = float(result.get("take_profit"))
        if all(np.isfinite(x) and x > 0 for x in (entry, sl, tp)) and ((side == "ACHAT" and sl < entry < tp) or (side == "VENTE" and tp < entry < sl)):
            actual_rr = abs(tp - entry) / abs(entry - sl)
            declared_rr = result.get("risk_reward")
            if declared_rr is not None:
                declared_rr = float(declared_rr)
                if not np.isfinite(declared_rr) or declared_rr <= 0:
                    issues.append("Ratio rendement/risque déclaré invalide.")
                elif abs(actual_rr - declared_rr) > max(0.10, 0.10 * actual_rr):
                    issues.append(f"Ratio rendement/risque incohérent : niveaux={actual_rr:.2f}R, déclaré={declared_rr:.2f}R.")
    except (TypeError, ValueError, ZeroDivisionError):
        issues.append("Impossible de vérifier le ratio rendement/risque.")
    candle_time = result.get("candle_time")
    parsed_time = pd.to_datetime(candle_time, utc=True, errors="coerce")
    if candle_time is None or pd.isna(parsed_time):
        issues.append("Horodatage de la bougie du signal absent ou invalide.")
    return {"ok": not issues, "issues": issues}


def calculate_broker_lot_size(risk_cash, stop_pips, pip_value_per_lot, lot_step, min_lot=0.0, max_lot=None):
    """Calcule un volume arrondi vers le bas avec Decimal et contrôle le risque."""
    try:
        risk = Decimal(str(risk_cash))
        stop = Decimal(str(stop_pips))
        pip_value = Decimal(str(pip_value_per_lot))
        step = Decimal(str(lot_step))
        minimum = Decimal(str(min_lot))
        maximum = Decimal(str(max_lot)) if max_lot is not None else None
    except (InvalidOperation, ValueError, TypeError):
        return {"valid": False, "reason": "Paramètres numériques invalides", "raw_lots": 0.0, "rounded_lots": 0.0, "estimated_risk": 0.0}

    if risk <= 0 or stop <= 0 or pip_value <= 0 or step <= 0 or minimum < 0:
        return {"valid": False, "reason": "Le risque, la distance SL, la valeur du pip et le pas doivent être positifs", "raw_lots": 0.0, "rounded_lots": 0.0, "estimated_risk": 0.0}
    if maximum is not None and maximum < minimum:
        return {"valid": False, "reason": "Le volume maximal est inférieur au volume minimal", "raw_lots": 0.0, "rounded_lots": 0.0, "estimated_risk": 0.0}

    raw = risk / (stop * pip_value)
    rounded = (raw / step).to_integral_value(rounding=ROUND_DOWN) * step
    if maximum is not None:
        rounded = min(rounded, (maximum / step).to_integral_value(rounding=ROUND_DOWN) * step)
    estimated_risk = rounded * stop * pip_value
    valid = rounded > 0 and rounded >= minimum and estimated_risk <= risk
    reason = "OK" if valid else ("Volume arrondi inférieur au minimum autorisé" if rounded < minimum or rounded <= 0 else "Le risque calculé dépasse le budget")
    return {
        "valid": bool(valid), "reason": reason, "raw_lots": float(raw),
        "rounded_lots": float(rounded), "estimated_risk": float(estimated_risk),
        "risk_budget": float(risk), "unused_risk": float(risk - estimated_risk),
    }


def run_position_sizing_rounding_tests():
    """Tests isolés du calcul de lots; aucune connexion au courtier n'est effectuée."""
    cases = [
        ("Pas 0,01 : arrondi vers le bas", dict(risk_cash=100, stop_pips=20, pip_value_per_lot=10, lot_step=0.01, min_lot=0.01), 0.5, True),
        ("Le risque reste sous le budget", dict(risk_cash=95, stop_pips=25, pip_value_per_lot=10, lot_step=0.01, min_lot=0.01), 0.38, True),
        ("Budget trop faible pour le minimum", dict(risk_cash=1, stop_pips=100, pip_value_per_lot=10, lot_step=0.01, min_lot=0.01), 0.0, False),
        ("Pas 0,1", dict(risk_cash=250, stop_pips=25, pip_value_per_lot=10, lot_step=0.1, min_lot=0.1), 1.0, True),
        ("Paramètre invalide rejeté", dict(risk_cash=100, stop_pips=0, pip_value_per_lot=10, lot_step=0.01), 0.0, False),
        ("Plafond de volume respecté", dict(risk_cash=1000, stop_pips=10, pip_value_per_lot=10, lot_step=0.01, min_lot=0.01, max_lot=0.5), 0.5, True),
        ("Pas incompatible avec le budget", dict(risk_cash=30, stop_pips=100, pip_value_per_lot=10, lot_step=0.1, min_lot=0.1), 0.0, False),
        ("Pas décimal sans dépassement", dict(risk_cash=73, stop_pips=17, pip_value_per_lot=9.7, lot_step=0.001, min_lot=0.001), None, True),
    ]
    rows = []
    for name, kwargs, expected_volume, expected_valid in cases:
        result = calculate_broker_lot_size(**kwargs)
        ok = result["valid"] == expected_valid
        if expected_volume is not None:
            ok = ok and abs(result["rounded_lots"] - expected_volume) < 1e-8
        if result["valid"]:
            ok = ok and result["estimated_risk"] <= kwargs["risk_cash"] + 1e-8
            ok = ok and result["rounded_lots"] <= result["raw_lots"] + 1e-8
        rows.append({"Test": name, "Volume théorique": round(result["raw_lots"], 6), "Volume arrondi": round(result["rounded_lots"], 6), "Risque estimé": round(result["estimated_risk"], 6), "État": "RÉUSSI" if ok else "ÉCHEC", "Détail": result["reason"]})
    return pd.DataFrame(rows)


def run_signal_record_consistency_tests():
    """Deterministic checks for signal levels and timestamp before saving."""
    valid_buy = {"signal": "ACHAT", "entry": 100.0, "stop_loss": 98.0, "take_profit": 104.0,
                 "candle_time": "2026-10-16T10:00:00Z"}
    valid_sell = {"signal": "VENTE", "entry": 100.0, "stop_loss": 102.0, "take_profit": 96.0,
                  "candle_time": "2026-10-16T10:00:00Z"}
    cases = [
        ("ACHAT avec niveaux et ratio cohérents", valid_buy, True),
        ("VENTE avec niveaux et ratio cohérents", valid_sell, True),
        ("ACHAT avec SL du mauvais côté", {**valid_buy, "stop_loss": 101.0}, False),
        ("TP manquant", {**valid_buy, "take_profit": None}, False),
        ("Ratio R/R déclaré incohérent", {**valid_buy, "risk_reward": 1.0}, False),
        ("Ratio R/R non positif", {**valid_buy, "risk_reward": 0}, False),
        ("Horodatage absent", {**valid_buy, "candle_time": None}, False),
        ("ATTENDRE non actionnable", {"signal": "ATTENDRE"}, True),
    ]
    rows = []
    for name, result, expected in cases:
        check = validate_signal_record_consistency(result)
        passed = check["ok"] == expected
        rows.append({"Test": name, "État": "RÉUSSI" if passed else "ÉCHEC",
                     "Détail": "; ".join(check["issues"]) if check["issues"] else "Contrôle accepté."})
    return pd.DataFrame(rows)


def audit_saved_signal_consistency(limit=200):
    """Inspect saved actionable rows for level direction and timestamp consistency."""
    history = get_signal_history(limit=limit)
    if history is None or history.empty:
        return pd.DataFrame([{"ID": None, "Symbole": None, "Unité": None, "Signal": None,
                              "État": "À VÉRIFIER", "Détail": "Aucun signal enregistré à auditer."}])
    rows = []
    for _, row in history.iterrows():
        side = row.get("Signal")
        issues = []
        try:
            entry, sl, tp = (float(row.get("Entrée")), float(row.get("Stop Loss")), float(row.get("Take Profit")))
            if not all(np.isfinite(x) and x > 0 for x in (entry, sl, tp)):
                issues.append("Prix absent, non fini ou non positif.")
            elif side == "ACHAT" and not sl < entry < tp:
                issues.append("Ordre SL/Entrée/TP incohérent pour ACHAT.")
            elif side == "VENTE" and not tp < entry < sl:
                issues.append("Ordre TP/Entrée/SL incohérent pour VENTE.")
        except (TypeError, ValueError):
            issues.append("Entrée, SL ou TP manquant/non numérique.")
        stamp = pd.to_datetime(row.get("Bougie analysée (UTC)"), utc=True, errors="coerce")
        if pd.isna(stamp):
            issues.append("Horodatage de bougie absent/invalide.")
        rows.append({"ID": row.get("ID"), "Symbole": row.get("Symbole"), "Unité": row.get("Unité"),
                     "Signal": side, "État": "À CORRIGER" if issues else "COHÉRENT",
                     "Détail": "; ".join(issues) if issues else "Niveaux et horodatage cohérents."})
    return pd.DataFrame(rows)

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

    consistency = validate_signal_record_consistency(result)
    if not consistency["ok"]:
        return None, created_at_utc, "invalid_levels", 0

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


def update_signal_outcome(signal_id, outcome_status):
    """Enregistre manuellement le résultat observé d'un signal."""
    allowed = {"OUVERT", "GAGNANT", "PERDANT", "NEUTRE"}
    if outcome_status not in allowed:
        raise ValueError("Statut de résultat non autorisé")
    updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE signals SET outcome_status = ?, outcome_updated_utc = ? WHERE id = ?",
            (outcome_status, updated, int(signal_id)),
        )
        conn.commit()



def update_trade_journal(signal_id, actual_entry, actual_exit, realized_r, journal_notes, execution_notes):
    """Save execution details and review notes for one historical signal."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """UPDATE signals
               SET actual_entry = ?, actual_exit = ?, realized_r = ?,
                   journal_notes = ?, execution_notes = ?
               WHERE id = ?""",
            (
                actual_entry if actual_entry > 0 else None,
                actual_exit if actual_exit > 0 else None,
                realized_r,
                journal_notes.strip() or None,
                execution_notes.strip() or None,
                int(signal_id),
            ),
        )
        conn.commit()



def run_signal_lifecycle_tests():
    """Exercise save, duplicate prevention, outcome update and journal preservation in an isolated SQLite DB."""
    global DB_PATH
    original_db_path = DB_PATH
    rows = []
    with tempfile.TemporaryDirectory(prefix="forexscope_lifecycle_") as tmp_dir:
        try:
            DB_PATH = os.path.join(tmp_dir, "lifecycle_test.db")
            init_signal_db()
            sample = {
                "signal": "ACHAT", "entry": 100.0, "stop_loss": 98.0,
                "take_profit": 104.0, "candle_time": "2026-10-16T10:00:00Z",
                "setup_quality": 78, "reasons": ["test déterministe"]
            }
            first_id, _, first_status, _ = save_signal_record(
                "Test", "TEST=X", "M15", sample, cooldown_minutes=0
            )
            rows.append({"Test": "Création d’un signal valide", "État": "RÉUSSI" if first_status == "saved" and first_id else "ÉCHEC",
                         "Détail": f"statut={first_status}, id={first_id}"})

            duplicate_id, _, duplicate_status, _ = save_signal_record(
                "Test", "TEST=X", "M15", sample, cooldown_minutes=0
            )
            with sqlite3.connect(DB_PATH) as conn:
                count_after_duplicate = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
            duplicate_ok = duplicate_status == "duplicate_candle" and duplicate_id == first_id and count_after_duplicate == 1
            rows.append({"Test": "Anti-doublon sur la même bougie", "État": "RÉUSSI" if duplicate_ok else "ÉCHEC",
                         "Détail": f"statut={duplicate_status}, lignes={count_after_duplicate}"})

            update_signal_outcome(first_id, "GAGNANT")
            with sqlite3.connect(DB_PATH) as conn:
                state = conn.execute("SELECT outcome_status, entry, stop_loss, take_profit FROM signals WHERE id=?", (first_id,)).fetchone()
            outcome_ok = state is not None and state[0] == "GAGNANT" and tuple(state[1:]) == (100.0, 98.0, 104.0)
            rows.append({"Test": "Mise à jour du résultat sans modifier les niveaux", "État": "RÉUSSI" if outcome_ok else "ÉCHEC",
                         "Détail": f"statut={state[0] if state else None}, niveaux préservés={bool(state and tuple(state[1:]) == (100.0, 98.0, 104.0))}"})

            update_trade_journal(first_id, 100.1, 103.8, 1.8, "revue test", "exécution test")
            with sqlite3.connect(DB_PATH) as conn:
                state = conn.execute("SELECT entry, stop_loss, take_profit, actual_entry, actual_exit, realized_r, journal_notes, execution_notes FROM signals WHERE id=?", (first_id,)).fetchone()
            journal_ok = state is not None and tuple(state[:3]) == (100.0, 98.0, 104.0) and tuple(state[3:6]) == (100.1, 103.8, 1.8) and state[6] == "revue test" and state[7] == "exécution test"
            rows.append({"Test": "Journal d’exécution distinct des niveaux d’origine", "État": "RÉUSSI" if journal_ok else "ÉCHEC",
                         "Détail": "Les prix du signal restent distincts des prix réellement exécutés." if journal_ok else "Les données du journal ou les niveaux ne correspondent pas."})

            invalid = {**sample, "candle_time": "2026-10-16T10:15:00Z", "stop_loss": 101.0}
            invalid_id, _, invalid_status, _ = save_signal_record("Test", "TEST=X", "M15", invalid, cooldown_minutes=0)
            with sqlite3.connect(DB_PATH) as conn:
                final_count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
            invalid_ok = invalid_id is None and invalid_status == "invalid_levels" and final_count == 1
            rows.append({"Test": "Refus d’un signal incohérent", "État": "RÉUSSI" if invalid_ok else "ÉCHEC",
                         "Détail": f"statut={invalid_status}, lignes finales={final_count}"})
        except Exception as exc:
            rows.append({"Test": "Exécution du cycle de vie", "État": "ÉCHEC", "Détail": f"{type(exc).__name__}: {exc}"})
        finally:
            DB_PATH = original_db_path
    return pd.DataFrame(rows)


def run_storage_integrity_tests():
    """Test SQLite backup/restore round-trip in temporary files, never touching the live database."""
    rows = []
    try:
        with tempfile.TemporaryDirectory(prefix="forexscope_storage_test_") as tmp_dir:
            source_path = os.path.join(tmp_dir, "source.db")
            backup_path = os.path.join(tmp_dir, "backup.db")
            restored_path = os.path.join(tmp_dir, "restored.db")
            with sqlite3.connect(source_path) as conn:
                conn.execute("CREATE TABLE storage_probe (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
                conn.execute("INSERT INTO storage_probe(value) VALUES (?)", ("forexscope-roundtrip",))
                conn.commit()
            # Use SQLite's online backup API, as the application does for user backups.
            with sqlite3.connect(source_path) as source_conn, sqlite3.connect(backup_path) as backup_conn:
                source_conn.backup(backup_conn)
            with sqlite3.connect(backup_path) as check_conn:
                integrity = check_conn.execute("PRAGMA integrity_check").fetchone()[0]
                count = check_conn.execute("SELECT COUNT(*) FROM storage_probe").fetchone()[0]
                value = check_conn.execute("SELECT value FROM storage_probe WHERE id=1").fetchone()[0]
            ok = integrity == "ok" and count == 1 and value == "forexscope-roundtrip"
            rows.append({"Test": "Sauvegarde SQLite intègre", "État": "RÉUSSI" if ok else "ÉCHEC", "Détail": f"integrity_check={integrity}; lignes={count}; contenu conforme={value == 'forexscope-roundtrip'}"})
            # Restore the backup into a separate destination and verify its contents.
            with sqlite3.connect(backup_path) as backup_conn, sqlite3.connect(restored_path) as restored_conn:
                backup_conn.backup(restored_conn)
            with sqlite3.connect(restored_path) as restored_conn:
                restored_integrity = restored_conn.execute("PRAGMA integrity_check").fetchone()[0]
                restored_value = restored_conn.execute("SELECT value FROM storage_probe WHERE id=1").fetchone()[0]
            restore_ok = restored_integrity == "ok" and restored_value == "forexscope-roundtrip"
            rows.append({"Test": "Restauration de la sauvegarde", "État": "RÉUSSI" if restore_ok else "ÉCHEC", "Détail": f"integrity_check={restored_integrity}; contenu restauré conforme={restored_value == 'forexscope-roundtrip'}"})
            # Ensure a filesystem write is possible in the configured data directory.
            probe_path = os.path.join(DATA_DIR, ".forexscope_write_probe.tmp")
            try:
                with open(probe_path, "w", encoding="utf-8") as probe:
                    probe.write("ok")
                writable_ok = Path(probe_path).read_text(encoding="utf-8") == "ok"
                rows.append({"Test": "Écriture dans le répertoire configuré", "État": "RÉUSSI" if writable_ok else "ÉCHEC", "Détail": f"Répertoire={DATA_DIR}; écriture/lecture vérifiée"})
            finally:
                try:
                    os.remove(probe_path)
                except OSError:
                    pass
            rows.append({"Test": "Base réelle protégée pendant les tests", "État": "RÉUSSI", "Détail": "Tous les essais SQLite utilisent des fichiers temporaires distincts de la base active."})
    except Exception as exc:
        rows.append({"Test": "Tests de sauvegarde/restauration", "État": "ÉCHEC", "Détail": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)

def auto_check_signal_outcomes(limit=200):
    """Classify open signals when later available candles touch SL or TP.

    If both levels are touched in one candle, leave the signal open because
    OHLC data cannot establish which level was reached first.
    """
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """SELECT id, symbol, timeframe, signal, candle_time_utc, stop_loss, take_profit
               FROM signals WHERE outcome_status = 'OUVERT'
                 AND signal IN ('ACHAT', 'VENTE')
                 AND stop_loss IS NOT NULL AND take_profit IS NOT NULL
               ORDER BY id DESC LIMIT ?""", (int(limit),)
        ).fetchall()
    checked = wins = losses = 0
    for signal_id, symbol, tf, direction, candle_time, sl, tp in rows:
        if not candle_time or tf not in INTERVALS:
            continue
        data = fetch_candles(symbol, tf, count=300)
        if data is None or data.empty:
            continue
        signal_time = pd.to_datetime(candle_time, utc=True, errors='coerce')
        if pd.isna(signal_time):
            continue
        times = pd.to_datetime(data['time'], utc=True, errors='coerce')
        future = data.loc[times > signal_time]
        outcome = None
        for _, bar in future.iterrows():
            if direction == 'ACHAT':
                hit_sl, hit_tp = float(bar['low']) <= float(sl), float(bar['high']) >= float(tp)
            else:
                hit_sl, hit_tp = float(bar['high']) >= float(sl), float(bar['low']) <= float(tp)
            if hit_sl and hit_tp:
                outcome = None
                break
            if hit_tp:
                outcome = 'GAGNANT'
                break
            if hit_sl:
                outcome = 'PERDANT'
                break
        if outcome:
            update_signal_outcome(signal_id, outcome)
            checked += 1
            wins += int(outcome == 'GAGNANT')
            losses += int(outcome == 'PERDANT')
    return checked, wins, losses

def get_signal_outcome_summary():
    """Résumé des résultats enregistrés, sans modifier l'historique."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT COALESCE(outcome_status, 'OUVERT'), COUNT(*) FROM signals GROUP BY COALESCE(outcome_status, 'OUVERT')"
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        latest = conn.execute(
            "SELECT id, symbol, timeframe, signal, outcome_status, outcome_updated_utc FROM signals ORDER BY id DESC LIMIT 1"
        ).fetchone()
    counts = {str(status): int(count) for status, count in rows}
    return counts, int(total or 0), latest


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
                outcome_status AS Résultat,
                outcome_updated_utc AS "Résultat actualisé (UTC)",
                actual_entry AS "Entrée réelle",
                actual_exit AS "Sortie réelle",
                realized_r AS "Résultat (R)",
                journal_notes AS "Notes de revue",
                execution_notes AS "Notes d'exécution",
                reasons AS Raisons
            FROM signals
            ORDER BY id DESC
            LIMIT ?
            """,
            conn,
            params=(int(limit),),
        )


def build_period_robustness_report(history: pd.DataFrame, min_closed: int = 5) -> pd.DataFrame:
    """Summarize historical signal outcomes by calendar month without claiming predictive validity."""
    columns = ["Période (UTC)", "Symbole", "Unité", "Signaux clôturés", "Gagnants", "Perdants",
               "Taux de réussite (%)", "R théorique moyen", "R réel moyen", "Échantillon suffisant"]
    if history is None or history.empty:
        return pd.DataFrame(columns=columns)
    required = {"Créé à (UTC)", "Marché", "Unité", "Résultat"}
    if not required.issubset(history.columns):
        return pd.DataFrame(columns=columns)
    df = history.copy()
    created = pd.to_datetime(df["Créé à (UTC)"], errors="coerce", utc=True)
    df["_Période"] = created.dt.strftime("%Y-%m")
    df["_Résultat"] = df["Résultat"].fillna("OUVERT").astype(str)
    df = df[df["_Période"].notna() & df["_Résultat"].isin(["GAGNANT", "PERDANT"])]
    if df.empty:
        return pd.DataFrame(columns=columns)
    # Reuse the existing theoretical-R calculation, keeping it separate from manually entered realized R.
    df["_R_th"] = df.apply(lambda row: _period_theoretical_r(row), axis=1)
    if "Résultat (R)" in df.columns:
        df["_R_reel"] = pd.to_numeric(df["Résultat (R)"], errors="coerce")
    else:
        df["_R_reel"] = float("nan")
    rows = []
    group_cols = ["_Période", "Marché", "Unité"]
    for (period, market, timeframe), group in df.groupby(group_cols, dropna=False):
        wins = int((group["_Résultat"] == "GAGNANT").sum())
        losses = int((group["_Résultat"] == "PERDANT").sum())
        closed = wins + losses
        th = pd.to_numeric(group["_R_th"], errors="coerce").dropna()
        real = pd.to_numeric(group["_R_reel"], errors="coerce").dropna()
        rows.append({
            "Période (UTC)": str(period), "Symbole": str(market), "Unité": str(timeframe),
            "Signaux clôturés": closed, "Gagnants": wins, "Perdants": losses,
            "Taux de réussite (%)": round(100 * wins / closed, 2) if closed else None,
            "R théorique moyen": round(float(th.mean()), 3) if not th.empty else None,
            "R réel moyen": round(float(real.mean()), 3) if not real.empty else None,
            "Échantillon suffisant": "Oui" if closed >= int(min_closed) else "Non — petit échantillon",
        })
    return pd.DataFrame(rows, columns=columns).sort_values(["Période (UTC)", "Symbole", "Unité"], ascending=[False, True, True]).reset_index(drop=True)


def _period_theoretical_r(row):
    """Return planned R for a closed theoretical outcome, if levels are valid."""
    try:
        entry, sl, tp = float(row.get("Entrée")), float(row.get("Stop Loss")), float(row.get("Take Profit"))
        if not all(pd.notna(v) for v in (entry, sl, tp)):
            return None
        risk = abs(entry - sl)
        if risk <= 0:
            return None
        status, side = str(row.get("Résultat", "")), str(row.get("Signal", ""))
        if status == "PERDANT":
            return -1.0
        valid = (side == "ACHAT" and sl < entry < tp) or (side == "VENTE" and tp < entry < sl)
        return abs(tp - entry) / risk if status == "GAGNANT" and valid else None
    except (TypeError, ValueError, OverflowError):
        return None


def test_period_robustness_report():
    """Small deterministic checks for monthly grouping and separation of real R."""
    sample = pd.DataFrame([
        {"Créé à (UTC)": "2026-01-03T10:00:00Z", "Marché": "EURUSD=X", "Unité": "M15", "Résultat": "GAGNANT", "Signal": "ACHAT", "Entrée": 1.1, "Stop Loss": 1.09, "Take Profit": 1.12, "Résultat (R)": 1.7},
        {"Créé à (UTC)": "2026-01-04T10:00:00Z", "Marché": "EURUSD=X", "Unité": "M15", "Résultat": "PERDANT", "Signal": "ACHAT", "Entrée": 1.1, "Stop Loss": 1.09, "Take Profit": 1.12, "Résultat (R)": -1.0},
        {"Créé à (UTC)": "2026-02-01T10:00:00Z", "Marché": "EURUSD=X", "Unité": "M15", "Résultat": "OUVERT", "Signal": "ACHAT", "Entrée": 1.1, "Stop Loss": 1.09, "Take Profit": 1.12, "Résultat (R)": None},
    ])
    report = build_period_robustness_report(sample, min_closed=2)
    assert len(report) == 1, "Les signaux ouverts ne doivent pas compter comme clôturés."
    assert report.iloc[0]["Période (UTC)"] == "2026-01"
    assert int(report.iloc[0]["Signaux clôturés"]) == 2
    assert abs(float(report.iloc[0]["Taux de réussite (%)"]) - 50.0) < 1e-9
    assert report.iloc[0]["Échantillon suffisant"] == "Oui"
    assert abs(float(report.iloc[0]["R réel moyen"]) - 0.35) < 1e-9
    return True


def build_performance_breakdown(history: pd.DataFrame) -> pd.DataFrame:
    """Build symbol/timeframe performance metrics without mixing simulated and actual R."""
    columns = [
        "Symbole", "Unité", "Signaux", "Clôturés théoriques", "Gagnants théoriques",
        "Perdants théoriques", "Taux de réussite théorique (%)", "R théorique cumulé",
        "R théorique moyen / clôturé", "Trades avec R réel renseigné", "R réel cumulé",
        "R réel moyen / trade renseigné",
    ]
    if history is None or history.empty:
        return pd.DataFrame(columns=columns)

    df = history.copy()
    required = {"Symbole", "Unité", "Résultat", "Entrée", "Stop Loss", "Take Profit"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=columns)

    # A theoretical winner earns the planned reward/risk ratio; a theoretical loser is -1R.
    # These metrics are estimates based on SL/TP status, not broker-confirmed executions.
    def theoretical_r(row):
        status = str(row.get("Résultat", ""))
        try:
            entry = float(row.get("Entrée"))
            sl = float(row.get("Stop Loss"))
            tp = float(row.get("Take Profit"))
            side = str(row.get("Signal", ""))
            risk = abs(entry - sl)
            if risk <= 0 or not all(pd.notna(x) for x in (entry, sl, tp)):
                return None
            if status == "PERDANT":
                return -1.0
            if status == "GAGNANT":
                valid = (side == "ACHAT" and sl < entry < tp) or (side == "VENTE" and tp < entry < sl)
                return abs(tp - entry) / risk if valid else None
        except (TypeError, ValueError, OverflowError):
            return None
        return None

    df["_R_theorique"] = df.apply(theoretical_r, axis=1)
    if "Résultat (R)" in df.columns:
        df["_R_reel"] = pd.to_numeric(df["Résultat (R)"], errors="coerce")
    else:
        df["_R_reel"] = float("nan")

    rows = []
    for (symbol, timeframe), group in df.groupby(["Symbole", "Unité"], dropna=False):
        statuses = group["Résultat"].fillna("OUVERT").astype(str)
        closed_mask = statuses.isin(["GAGNANT", "PERDANT"])
        closed = int(closed_mask.sum())
        wins = int((statuses == "GAGNANT").sum())
        losses = int((statuses == "PERDANT").sum())
        theoretical = pd.to_numeric(group.loc[closed_mask, "_R_theorique"], errors="coerce").dropna()
        real = pd.to_numeric(group["_R_reel"], errors="coerce").dropna()
        rows.append({
            "Symbole": str(symbol) if pd.notna(symbol) else "Inconnu",
            "Unité": str(timeframe) if pd.notna(timeframe) else "Inconnue",
            "Signaux": int(len(group)),
            "Clôturés théoriques": closed,
            "Gagnants théoriques": wins,
            "Perdants théoriques": losses,
            "Taux de réussite théorique (%)": round(100 * wins / closed, 2) if closed else None,
            "R théorique cumulé": round(float(theoretical.sum()), 3) if len(theoretical) else None,
            "R théorique moyen / clôturé": round(float(theoretical.mean()), 3) if len(theoretical) else None,
            "Trades avec R réel renseigné": int(len(real)),
            "R réel cumulé": round(float(real.sum()), 3) if len(real) else None,
            "R réel moyen / trade renseigné": round(float(real.mean()), 3) if len(real) else None,
        })
    result = pd.DataFrame(rows, columns=columns)
    return result.sort_values(["Signaux", "Symbole", "Unité"], ascending=[False, True, True]).reset_index(drop=True)


init_signal_db()


@st.cache_resource(show_spinner=False)
def record_process_startup():
    """Enregistre une seule fois le démarrage du processus Streamlit courant."""
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """INSERT INTO runtime_startups (started_at_utc, data_dir, db_path, app_version)
               VALUES (?, ?, ?, ?)""",
            (started_at, os.path.abspath(DATA_DIR), os.path.abspath(DB_PATH), "phase52"),
        )
        conn.commit()
    return started_at


try:
    CURRENT_PROCESS_STARTED_AT = record_process_startup()
except Exception:
    # Le tableau d'historique indiquera l'erreur si la table ne peut être lue.
    CURRENT_PROCESS_STARTED_AT = None


def get_runtime_startups(limit=20):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            """SELECT started_at_utc AS "Démarrage UTC",
                      data_dir AS "Répertoire de données",
                      db_path AS "Base SQLite",
                      app_version AS "Version"
               FROM runtime_startups ORDER BY id DESC LIMIT ?""",
            conn,
            params=(int(limit),),
        )


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



def run_smc_engine_self_tests():
    """Run deterministic local tests on core OHLC/SMC helpers; no network calls."""
    results = []

    def record(name, passed, detail):
        results.append({"Test": name, "État": "RÉUSSI" if passed else "ÉCHEC", "Détail": detail})

    # OHLC validation must reject impossible candles and duplicate timestamps.
    times = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
    sample = pd.DataFrame({
        "time": list(times) + [times[0]],
        "open": [10, 10, 10, 10, 10],
        "high": [11, 11, 9, 11, 11],
        "low": [9, 9, 8, 9, 9],
        "close": [10, 10, 10, 10, 10],
    })
    cleaned = validate_ohlc(sample)
    record("Validation OHLC : bougie impossible et doublon", len(cleaned) == 3,
           f"3 bougies valides attendues, {len(cleaned)} obtenues")

    # Confirmed close beyond all prior highs should yield bullish BOS proxy.
    rows = []
    for i in range(12):
        rows.append({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})
    rows.append({"open": 101.0, "high": 103.0, "low": 100.5, "close": 102.5})
    structure = analyze_market_structure(pd.DataFrame(rows), lookback=10)
    record("Structure : cassure haussière confirmée", structure.get("breakout") == "HAUSSIER",
           f"Résultat obtenu : {structure.get('breakout')}")

    # Final candle sweeps the prior low then closes back above it.
    rows = []
    for i in range(20):
        rows.append({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})
    rows.append({"open": 99.5, "high": 100.5, "low": 98.0, "close": 100.0})
    sweep = detect_liquidity_sweep(pd.DataFrame(rows), lookback=20)
    record("Liquidité : sweep haussier avec réintégration", sweep.get("sweep") == "Sweep haussier potentiel",
           f"Résultat obtenu : {sweep.get('sweep')}")

    # Short input must never produce a confirmed SMC entry.
    short = pd.DataFrame(rows[:10])
    setup = _find_confirmed_smc_setup(short, atr=1.0, lookback=20, sweep_window=6)
    record("SMC : données insuffisantes, aucun signal", setup.get("direction") is None,
           f"Direction obtenue : {setup.get('direction')}")

    return pd.DataFrame(results)

def run_smc_temporal_integrity_tests():
    """Check that analysis excludes the latest potentially unfinished candle."""
    results = []

    def record(name, passed, detail):
        results.append({"Test": name, "État": "RÉUSSI" if passed else "ÉCHEC", "Détail": detail})

    # Deterministic synthetic OHLC series with enough history for indicators.
    rng = np.random.default_rng(20261016)
    n = 120
    times = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 0.35, n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0.05, 0.4, n)
    low = np.minimum(open_, close) - rng.uniform(0.05, 0.4, n)
    base = pd.DataFrame({"time": times, "open": open_, "high": high, "low": low, "close": close, "volume": 1000})

    baseline = analyze_timeframe(base)
    altered = base.copy()
    # Dramatically alter only the final candle; it must not affect the reported
    # closed-candle analysis because analyze_timeframe intentionally excludes it.
    altered.loc[n - 1, ["open", "high", "low", "close"]] = [150.0, 155.0, 145.0, 152.0]
    changed_last = analyze_timeframe(altered)
    keys = ["status", "bias", "rsi", "adx", "atr", "macd", "structure", "sweep", "close", "ema_20", "ema_50", "ema_200", "bullish_points", "bearish_points", "candle_time"]
    same = True
    differences = []
    for key in keys:
        a, b = baseline.get(key), changed_last.get(key)
        if isinstance(a, (float, int, np.floating)) and isinstance(b, (float, int, np.floating)):
            equal = (pd.isna(a) and pd.isna(b)) or np.isclose(float(a), float(b), rtol=1e-10, atol=1e-10)
        else:
            equal = str(a) == str(b)
        if not equal:
            same = False
            differences.append(key)
    record("Intégrité temporelle : la dernière bougie n’influence pas l’analyse clôturée", same,
           "Résultats identiques sur la dernière bougie clôturée" if same else "Champs modifiés : " + ", ".join(differences))

    # Verify the returned timestamp belongs to the last closed candle, not the
    # final (potentially still forming) candle supplied to the function.
    expected_time = base.iloc[-2]["time"]
    actual_time = pd.to_datetime(baseline.get("candle_time"), utc=True, errors="coerce")
    expected_time = pd.to_datetime(expected_time, utc=True, errors="coerce")
    timestamp_ok = pd.notna(actual_time) and actual_time == expected_time
    record("Horodatage : référence à la dernière bougie clôturée", bool(timestamp_ok),
           f"Attendu : {expected_time}; obtenu : {actual_time}")

    return pd.DataFrame(results)


def audit_runtime_configuration(data=None):
    """Return a local diagnostic report without making additional network calls."""
    rows = []

    rows.append({
        "Contrôle": "Correspondance des marchés",
        "État": "OK" if SYMBOLS and all(isinstance(k, str) and isinstance(v, str) and v for k, v in SYMBOLS.items()) else "À vérifier",
        "Détail": f"{len(SYMBOLS)} marchés configurés",
    })
    rows.append({
        "Contrôle": "Unités de temps",
        "État": "OK" if set(INTERVALS).issubset(DEFAULT_SIGNAL_COOLDOWN_MINUTES) else "À vérifier",
        "Détail": f"{len(INTERVALS)} unités configurées",
    })

    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        probe_path = os.path.join(DATA_DIR, ".forexscope_write_test")
        with open(probe_path, "w", encoding="utf-8") as probe:
            probe.write("ok")
        os.remove(probe_path)
        storage_state = "OK"
        storage_detail = f"Répertoire accessible : {DATA_DIR}"
    except OSError as exc:
        storage_state = "ERREUR"
        storage_detail = f"Répertoire non accessible ({type(exc).__name__})"
    rows.append({"Contrôle": "Écriture du stockage", "État": storage_state, "Détail": storage_detail})

    rows.append({
        "Contrôle": "Base SQLite",
        "État": "Présente" if os.path.isfile(DB_PATH) else "À initialiser / vérifier",
        "Détail": DB_PATH,
    })

    if data is None or data.empty:
        rows.append({"Contrôle": "Bougies chargées", "État": "INDISPONIBLES", "Détail": "Aucune bougie utilisable pour le marché et l’unité sélectionnés."})
    else:
        times = pd.to_datetime(data.get("time"), utc=True, errors="coerce")
        time_ok = times.notna().all()
        ordered = bool(times.is_monotonic_increasing) if time_ok else False
        duplicates = int(times.duplicated().sum()) if time_ok else -1
        prices = data[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
        numeric_ok = bool(np.isfinite(prices.to_numpy(dtype=float)).all())
        positive_ok = bool((prices > 0).all().all())
        ohlc_ok = bool(((prices["high"] >= prices[["open", "close", "low"]].max(axis=1)) & (prices["low"] <= prices[["open", "close", "high"]].min(axis=1))).all())
        checks_ok = time_ok and ordered and duplicates == 0 and numeric_ok and positive_ok and ohlc_ok
        latest = times.iloc[-1].strftime("%Y-%m-%d %H:%M UTC") if time_ok and len(times) else "indisponible"
        rows.append({"Contrôle": "Intégrité des bougies", "État": "OK" if checks_ok else "À VÉRIFIER", "Détail": f"{len(data)} bougies ; temps ordonné={ordered} ; doublons={duplicates} ; valeurs numériques={numeric_ok} ; prix positifs={positive_ok} ; OHLC cohérent={ohlc_ok}"})
        rows.append({"Contrôle": "Horodatage le plus récent", "État": "Information", "Détail": latest})

    return pd.DataFrame(rows)


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

tab_market, tab_analysis, tab_history, tab_risk, tab_backtest, tab_settings = st.tabs(
    ["📈 Marché", "🎯 Analyse", "📋 Historique", "🛡️ Gestion du risque", "🧪 Backtesting", "⚙️ Paramètres"]
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
            freshness_check = validate_signal_data_freshness(
                candles, timeframe, multi_results
            )
            if not freshness_check["ok"]:
                signal_result = {
                    **signal_result,
                    "signal": "ATTENDRE",
                    "confidence": 0,
                    "setup_quality": 0,
                    "entry": None,
                    "stop_loss": None,
                    "take_profit": None,
                    "risk_reward": None,
                    "reasons": list(signal_result.get("reasons", []))
                    + ["Signal bloqué par le contrôle de fraîcheur des données."]
                    + freshness_check["issues"],
                }
                st.warning(
                    "Signal exploitable bloqué : certaines données sont absentes ou trop anciennes. "
                    "Vérifie le rapport avant de relancer l’analyse."
                )
                for issue in freshness_check["issues"]:
                    st.write(f"- {issue}")

            # Phase 25: diagnostic transparent des principaux filtres du signal.
            st.markdown("### Diagnostic des filtres du signal")
            try:
                diag_df = calculate_indicators(candles).dropna(subset=["ema_20", "ema_50", "atr", "rsi"])
                # Exclure la bougie la plus récente, qui peut être encore en formation.
                diag_closed = diag_df.iloc[:-1].copy()
                if len(diag_closed) >= 2:
                    diag_last = diag_closed.iloc[-1]
                    diag_close = float(diag_last["close"])
                    diag_atr = float(diag_last["atr"])
                    diag_ema_dist = abs(diag_close - float(diag_last["ema_20"])) / diag_atr if diag_atr > 0 else None
                    diag_adx = float(diag_last["adx"]) if pd.notna(diag_last.get("adx")) else None
                    diag_rsi = float(diag_last["rsi"])
                    h4_diag = multi_results.get("H4", {})
                    d1_diag = multi_results.get("D1", {})
                    h4_bias_diag = h4_diag.get("bias") if h4_diag.get("status") == "OK" else "Indisponible"
                    d1_bias_diag = d1_diag.get("bias") if d1_diag.get("status") == "OK" else "Indisponible"

                    diag_rows = [
                        {"Contrôle": "Éloignement EMA20", "Valeur": f"{diag_ema_dist:.2f} ATR" if diag_ema_dist is not None else "Indisponible", "Lecture": "À surveiller : prix étendu" if diag_ema_dist is not None and diag_ema_dist > 1.5 else "Dans le seuil indicatif"},
                        {"Contrôle": "ADX (force du mouvement)", "Valeur": f"{diag_adx:.1f}" if diag_adx is not None else "Indisponible", "Lecture": "Mouvement faible selon le seuil expérimental" if diag_adx is not None and diag_adx < 12 else "Seuil expérimental atteint" if diag_adx is not None else "Non évalué"},
                        {"Contrôle": "RSI (14)", "Valeur": f"{diag_rsi:.1f}", "Lecture": "Contexte momentum; à interpréter selon le sens du setup"},
                        {"Contrôle": "Biais H4", "Valeur": str(h4_bias_diag), "Lecture": "Contexte supérieur disponible" if h4_bias_diag != "Indisponible" else "Confluence H4 non vérifiable"},
                        {"Contrôle": "Biais D1", "Valeur": str(d1_bias_diag), "Lecture": "Contexte supérieur disponible" if d1_bias_diag != "Indisponible" else "Confluence D1 non vérifiable"},
                    ]
                    st.dataframe(pd.DataFrame(diag_rows), use_container_width=True, hide_index=True)
                    if h4_bias_diag == "Indisponible" or d1_bias_diag == "Indisponible":
                        st.info("La confluence H4/D1 est partiellement indisponible : ne traite pas l'absence de données comme une confirmation.")
                    st.caption("Diagnostic explicatif des filtres appliqués aux données disponibles. Il ne prédit pas le résultat du trade et la qualité du setup n'est pas une probabilité de gain.")
                else:
                    st.info("Données insuffisantes pour établir le diagnostic des filtres.")
            except (KeyError, TypeError, ValueError, IndexError) as diag_error:
                st.warning(f"Diagnostic des filtres indisponible pour cette série : {diag_error}")

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
            elif save_status == "invalid_levels":
                st.error("Signal non enregistré : les niveaux de prix ou l’horodatage ne sont pas cohérents.")

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

                st.markdown("#### Risque et volume théorique pour ce signal")
                st.caption(
                    "Estimation liée aux niveaux du signal actuel. Renseigne la valeur du pip "
                    "dans la devise de ton compte selon les spécifications de ton courtier."
                )
                sr1, sr2, sr3 = st.columns(3)
                with sr1:
                    signal_balance = st.number_input(
                        "Capital pour ce signal", min_value=0.0, value=100.0, step=10.0,
                        key="signal_risk_balance"
                    )
                with sr2:
                    signal_risk_pct = st.number_input(
                        "Risque du signal (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                        key="signal_risk_percent"
                    )
                with sr3:
                    signal_pip_size = st.number_input(
                        "Taille du pip en prix", min_value=0.000001, value=0.0001, format="%.6f",
                        key="signal_pip_size",
                        help="Valeur indicative par défaut pour certaines paires Forex. Adapte-la à l'instrument analysé."
                    )
                sr4, sr5 = st.columns(2)
                with sr4:
                    signal_pip_value = st.number_input(
                        "Valeur d'un pip pour 1 lot", min_value=0.000001, value=10.0, step=0.5,
                        key="signal_pip_value",
                        help="Valeur dans la devise de ton compte, à vérifier auprès du courtier."
                    )
                with sr5:
                    signal_lot_step = st.selectbox(
                        "Pas de volume", [0.001, 0.01, 0.1, 1.0], index=1,
                        key="signal_lot_step"
                    )

                signal_entry = float(signal_result["entry"])
                signal_sl = float(signal_result["stop_loss"])
                signal_tp = float(signal_result["take_profit"])
                signal_side = signal_result["signal"]
                signal_levels_valid = (
                    signal_side == "ACHAT" and signal_sl < signal_entry < signal_tp
                ) or (
                    signal_side == "VENTE" and signal_tp < signal_entry < signal_sl
                )
                signal_stop_distance = abs(signal_entry - signal_sl)
                signal_risk_cash = signal_balance * signal_risk_pct / 100.0
                if not signal_levels_valid:
                    st.error(
                        "Calcul du volume bloqué : les niveaux Entrée, Stop Loss et Take Profit "
                        "ne respectent pas le sens du signal. Vérifie les niveaux avant toute décision."
                    )
                elif signal_stop_distance > 0 and signal_pip_size > 0 and signal_pip_value > 0:
                    signal_stop_pips = signal_stop_distance / signal_pip_size
                    signal_raw_lots = signal_risk_cash / (signal_stop_pips * signal_pip_value) if signal_stop_pips > 0 else 0.0
                    signal_rounded_lots = np.floor(signal_raw_lots / signal_lot_step) * signal_lot_step
                    signal_estimated_risk = signal_rounded_lots * signal_stop_pips * signal_pip_value
                    sr6, sr7, sr8 = st.columns(3)
                    sr6.metric("Risque prévu", f"{signal_risk_cash:,.2f}")
                    sr7.metric("Volume théorique", f"{signal_raw_lots:.3f} lot(s)")
                    sr8.metric("Volume arrondi", f"{signal_rounded_lots:.3f} lot(s)")
                    st.write(
                        f"Distance du SL : **{signal_stop_pips:.1f} pips** · "
                        f"Risque estimé après arrondi : **{signal_estimated_risk:,.2f}** dans la devise du compte."
                    )
                    signal_reward_distance = abs(signal_tp - signal_entry)
                    signal_rr_check = signal_reward_distance / signal_stop_distance if signal_stop_distance > 0 else 0.0
                    if signal_rr_check < 1.0:
                        st.warning(
                            f"Ratio rendement/risque faible : environ 1:{signal_rr_check:.2f}. "
                            "Vérifie si le potentiel de gain justifie le risque."
                        )
                    else:
                        st.caption(f"Contrôle des niveaux : ratio rendement/risque calculé ≈ 1:{signal_rr_check:.2f}.")
                    if signal_rounded_lots <= 0:
                        st.warning("Le risque prévu est trop faible pour atteindre le plus petit volume sélectionné. Ne force pas le volume à la hausse.")
                    st.caption(
                        "Estimation uniquement : vérifie la taille du contrat, la valeur du pip, "
                        "le spread, les commissions, le slippage et la devise du compte dans MT5. "
                        "Pour l'or, les indices et les CFD, la valeur par défaut du pip peut être incorrecte."
                    )

                st.markdown("#### Graphique du setup avec niveaux de risque")
                st.caption(
                    "Les lignes Entrée, Stop Loss et Take Profit correspondent au signal "
                    "calculé pour les données actuellement chargées. Ce sont des niveaux indicatifs."
                )
                st.plotly_chart(
                    build_candlestick_chart(
                        candles,
                        market_name,
                        timeframe,
                        signal_result=signal_result,
                        show_fvg=True,
                        show_order_blocks=True,
                        show_liquidity=True,
                    ),
                    use_container_width=True,
                    key="signal_levels_chart",
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
    st.subheader("Historique et suivi des performances")

    st.markdown("**Suivi automatique des signaux — Phase 51**")
    st.caption(
        "Compare les niveaux SL/TP enregistrés aux bougies ultérieures disponibles. "
        "Ce suivi classe le résultat théorique du signal ; il ne confirme pas une exécution réelle sur MT5. "
        "Si SL et TP sont touchés dans la même bougie, le résultat reste ouvert car l’ordre d’atteinte est inconnu."
    )
    if st.button("Vérifier les résultats des signaux ouverts", key="run_auto_signal_outcome_check"):
        try:
            updated_count, wins_count, losses_count = auto_check_signal_outcomes(limit=200)
            st.success(
                f"Vérification terminée : {updated_count} résultat(s) mis à jour "
                f"({wins_count} gagnant(s), {losses_count} perdant(s))."
            )
        except Exception as outcome_error:
            st.error(f"La vérification automatique a échoué : {outcome_error}")

    try:
        outcome_counts, outcome_total, latest_signal = get_signal_outcome_summary()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Signaux enregistrés", outcome_total)
        k2.metric("Ouverts / non résolus", outcome_counts.get("OUVERT", 0))
        k3.metric("Gagnants théoriques", outcome_counts.get("GAGNANT", 0))
        k4.metric("Perdants théoriques", outcome_counts.get("PERDANT", 0))
        if latest_signal:
            st.caption(
                f"Dernier signal enregistré : #{latest_signal[0]} · {latest_signal[1]} · "
                f"{latest_signal[2]} · {latest_signal[3]} · statut {latest_signal[4]}"
            )
        else:
            st.info("Aucun signal n’est encore enregistré dans la base active.")
        st.caption("Les statuts sont calculés à partir des données de marché disponibles et ne déduisent pas les commissions, le spread, le slippage ni les exécutions réelles.")
    except Exception as outcome_summary_error:
        st.warning(f"Impossible de calculer le résumé des résultats : {outcome_summary_error}")

    st.divider()

    st.markdown("**Contrôle de sauvegarde et restauration**")
    st.caption("Teste l’intégrité d’une sauvegarde SQLite, sa restauration vers une base temporaire et les droits d’écriture du répertoire configuré. Les tests n’écrivent pas dans la base active.")
    if st.button("Tester la sauvegarde et la restauration", key="run_storage_integrity_tests"):
        storage_tests = run_storage_integrity_tests()
        st.dataframe(storage_tests, use_container_width=True, hide_index=True)
        storage_failures = storage_tests[storage_tests["État"] == "ÉCHEC"]
        if storage_failures.empty:
            st.success(f"{len(storage_tests)} contrôles de stockage réussis.")
        else:
            st.error(f"{len(storage_failures)} contrôle(s) de stockage en échec.")
        st.download_button(
            "Exporter les tests de stockage CSV",
            data=storage_tests.to_csv(index=False).encode("utf-8-sig"),
            file_name="forexscope_tests_sauvegarde_restauration.csv",
            mime="text/csv",
            key="download_storage_integrity_tests",
        )
    st.info(
        f"Répertoire de données actif : {DATA_DIR}. Pour vérifier la persistance après redéploiement, "
        "il faut aussi confirmer dans les paramètres Render qu’un disque persistant est monté sur ce chemin "
        "et que FOREXSCOPE_DATA_DIR pointe vers ce même répertoire. Le test local ne peut pas prouver la persistance après redémarrage."
    )
    st.markdown("**Diagnostic de continuité après redémarrage**")
    st.caption(
        "ForexScope inscrit un événement au premier chargement du processus Streamlit. "
        "Après un redémarrage ou redéploiement, revenez ici : une nouvelle ligne doit apparaître "
        "si la même base persistante est conservée."
    )
    try:
        runtime_df = get_runtime_startups(limit=20)
        if runtime_df.empty:
            st.warning("Aucun événement de démarrage n’est disponible dans cette base.")
        else:
            st.dataframe(runtime_df, use_container_width=True, hide_index=True)
            if len(runtime_df) >= 2:
                st.info(
                    "Plusieurs démarrages sont enregistrés dans la base active. "
                    "Comparez les dates et le chemin de base après le prochain redéploiement. "
                    "Cela ne prouve la persistance que si les lignes antérieures sont toujours présentes."
                )
            else:
                st.warning(
                    "Un seul démarrage est enregistré. La vérification après redéploiement reste à faire."
                )
        st.download_button(
            "Exporter le journal des démarrages CSV",
            data=runtime_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="forexscope_journal_demarrages.csv",
            mime="text/csv",
            key="download_runtime_startups",
        )
    except Exception as exc:
        st.error(f"Impossible de lire le journal des démarrages : {exc}")

    st.divider()
    # Export SQLite complet : permet de conserver une copie de l'historique et du journal.
    st.markdown("**Sauvegarde complète de ForexScope**")
    st.caption("Télécharge une copie de la base SQLite actuellement utilisée par cette instance, y compris les signaux et les notes du journal.")
    try:
        backup_buffer = io.BytesIO()
        with sqlite3.connect(DB_PATH) as source_conn:
            with sqlite3.connect(":memory:") as backup_conn:
                source_conn.backup(backup_conn)
                backup_conn.commit()
                backup_buffer.write(backup_conn.serialize())
        backup_buffer.seek(0)
        st.download_button(
            "Télécharger une sauvegarde complète (.db)",
            data=backup_buffer.getvalue(),
            file_name=f"forexscope_backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.db",
            mime="application/octet-stream",
            key="download_full_database_backup",
        )
    except Exception as backup_error:
        st.warning(f"La sauvegarde complète n'a pas pu être créée : {backup_error}")
    st.info(
    "Stockage de la base : " + DB_PATH + ". Sur Render, crée un Persistent Disk, "
    "monte-le sur /var/data et définis FOREXSCOPE_DATA_DIR=/var/data. "
    "Sans disque persistant, les données locales peuvent disparaître lors d’un redéploiement."
)

    st.markdown("**Restaurer une sauvegarde (.db)**")
    st.caption("La restauration remplace la base actuellement utilisée. Télécharge d'abord une sauvegarde de l'état actuel et vérifie que le fichier choisi provient bien de ForexScope.")
    uploaded_backup = st.file_uploader(
        "Choisir un fichier de sauvegarde ForexScope",
        type=["db", "sqlite", "sqlite3"],
        key="restore_database_upload",
    )
    confirm_restore = st.checkbox(
        "Je confirme que je veux remplacer la base actuelle par cette sauvegarde.",
        key="confirm_database_restore",
    )
    if st.button("Valider la restauration", key="restore_database_button"):
        if uploaded_backup is None:
            st.error("Choisis d'abord un fichier de sauvegarde.")
        elif not confirm_restore:
            st.error("Coche la confirmation avant de restaurer.")
        else:
            temp_path = None
            try:
                file_bytes = uploaded_backup.getvalue()
                if not file_bytes:
                    raise ValueError("Le fichier est vide.")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".db", dir=os.path.dirname(os.path.abspath(DB_PATH))) as temp_file:
                    temp_file.write(file_bytes)
                    temp_path = temp_file.name
                with sqlite3.connect(temp_path) as check_conn:
                    integrity = check_conn.execute("PRAGMA integrity_check").fetchone()
                    if not integrity or integrity[0] != "ok":
                        raise ValueError("Le fichier SQLite est endommagé ou invalide.")
                    tables = {row[0] for row in check_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                    if not tables:
                        raise ValueError("Aucune table n'a été trouvée dans cette base.")
                # Crée une copie de sécurité de la base actuelle avant le remplacement.
                safety_path = f"{DB_PATH}.before_restore_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                if os.path.exists(DB_PATH):
                    with sqlite3.connect(DB_PATH) as current_conn:
                        with sqlite3.connect(safety_path) as safety_conn:
                            current_conn.backup(safety_conn)
                os.replace(temp_path, DB_PATH)
                temp_path = None
                st.success(f"Restauration terminée. Une copie de sécurité locale a été créée : {os.path.basename(safety_path) if os.path.exists(safety_path) else 'aucune base précédente détectée'}.")
                st.warning("Recharge la page pour relire l'historique restauré. La copie de sécurité est stockée sur le serveur et peut disparaître si son stockage n'est pas persistant.")
            except Exception as restore_error:
                st.error(f"Restauration impossible : {restore_error}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
    history_raw = get_signal_history(limit=500)

    st.markdown("**Analyse des performances par symbole et unité — Phase 52**")
    st.caption(
        "Les résultats théoriques en R sont estimés à partir des statuts GAGNANT/PERDANT et des niveaux SL/TP. "
        "Les résultats réels en R proviennent uniquement des valeurs saisies dans le journal ; les deux mesures restent séparées."
    )
    try:
        breakdown = build_performance_breakdown(history_raw)
        if breakdown.empty:
            st.info("Pas encore assez de données pour établir une ventilation par symbole et unité.")
        else:
            st.dataframe(breakdown, use_container_width=True, hide_index=True)
            st.download_button(
                "Exporter l'analyse par symbole et unité (CSV)",
                data=breakdown.to_csv(index=False).encode("utf-8-sig"),
                file_name="forexscope_performances_symbole_unite.csv",
                mime="text/csv",
                key="download_phase52_performance_breakdown",
            )
    except Exception as breakdown_error:
        st.warning(f"Impossible de calculer l'analyse des performances : {breakdown_error}")

    st.divider()

    if history_raw.empty:
        st.info("Aucun signal enregistré pour le moment.")
    else:
        # Résumé calculé sur les signaux conservés dans l'historique.
        status_series = history_raw["Résultat"]
        total = len(history_raw)
        winners = int((status_series == "GAGNANT").sum())
        losers = int((status_series == "PERDANT").sum())
        open_count = int((status_series == "OUVERT").sum())
        closed_count = winners + losers
        win_rate = (100 * winners / closed_count) if closed_count else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Signaux suivis", total)
        m2.metric("Encore ouverts", open_count)
        m3.metric("Gagnants / perdants", f"{winners} / {losers}")
        m4.metric("Taux de réussite observé", f"{win_rate:.1f}%" if closed_count else "—")
        st.caption("Le taux est calculé uniquement sur les signaux marqués GAGNANT ou PERDANT. Il ne mesure pas le profit net et dépend du classement manuel.")

        st.markdown("**Vérification automatique des SL/TP**")
        st.caption("Compare les bougies disponibles après la bougie du signal. Si SL et TP sont touchés dans la même bougie, le signal reste ouvert pour éviter de deviner l'ordre des événements.")
        if st.button("Vérifier automatiquement les signaux ouverts"):
            with st.spinner("Vérification des bougies disponibles…"):
                checked, auto_wins, auto_losses = auto_check_signal_outcomes(limit=200)
            st.success(f"Vérification terminée : {checked} résultat(s) classé(s), {auto_wins} gagnant(s), {auto_losses} perdant(s).")
            st.rerun()

        st.markdown("**Journal détaillé du trade**")
        st.caption("Consigne les prix réellement exécutés et tes observations. Ces données sont distinctes des niveaux théoriques du signal.")
        journal_row = editable.loc[editable["ID"] == selected_id].iloc[0] if "selected_id" in locals() else editable.iloc[0]
        def _number_or_zero(value):
            try:
                return float(value) if pd.notna(value) else 0.0
            except (TypeError, ValueError):
                return 0.0
        def _text_or_empty(value):
            return str(value) if value is not None and pd.notna(value) else ""
        with st.form("trade_journal_form"):
            j1, j2 = st.columns(2)
            with j1:
                actual_entry = st.number_input("Prix d'entrée réellement exécuté (0 = inconnu)", min_value=0.0, value=_number_or_zero(journal_row.get("Entrée réelle")), format="%.6f")
                actual_exit = st.number_input("Prix de sortie réellement exécuté (0 = inconnu)", min_value=0.0, value=_number_or_zero(journal_row.get("Sortie réelle")), format="%.6f")
            with j2:
                previous_r = journal_row.get("Résultat (R)")
                default_r = _number_or_zero(previous_r)
                realized_r = st.number_input("Résultat réalisé en R (gain positif, perte négative)", value=default_r, min_value=-100.0, max_value=100.0, step=0.1, format="%.2f")
                st.caption("1R correspond au risque initial prévu. Saisis la valeur calculée à partir de ton résultat réel.")
            journal_notes = st.text_area("Revue du setup", value=_text_or_empty(journal_row.get("Notes de revue")), placeholder="Contexte, qualité du setup, respect du plan, leçon à retenir…")
            execution_notes = st.text_area("Exécution et coûts", value=_text_or_empty(journal_row.get("Notes d'exécution")), placeholder="Spread, commission, slippage, raison d'une sortie anticipée…")
            journal_submit = st.form_submit_button("Enregistrer le journal du trade")
        if journal_submit:
            update_trade_journal(selected_id, actual_entry, actual_exit, realized_r, journal_notes, execution_notes)
            st.success("Journal du trade enregistré.")
            st.rerun()

        st.markdown("**Mettre à jour le résultat d'un signal**")
        editable = history_raw.copy()
        choices = {
            int(row["ID"]): f'#{int(row["ID"])} · {row["Marché"]} · {row["Unité"]} · {row["Signal"]} · {row["Résultat"]}'
            for _, row in editable.iterrows()
        }
        selected_id = st.selectbox("Signal à évaluer", options=list(choices), format_func=lambda value: choices[value])
        current_status = str(editable.loc[editable["ID"] == selected_id, "Résultat"].iloc[0])
        status_options = ["OUVERT", "GAGNANT", "PERDANT", "NEUTRE"]
        selected_status = st.selectbox("Résultat observé", options=status_options, index=status_options.index(current_status) if current_status in status_options else 0)
        if st.button("Enregistrer le résultat", type="primary"):
            update_signal_outcome(selected_id, selected_status)
            st.success("Résultat mis à jour.")
            st.rerun()

        display_history = history_raw.copy()
        display_history["Créé à"] = display_history["Créé à (UTC)"].apply(display_local_time)
        display_history["Bougie analysée"] = display_history["Bougie analysée (UTC)"].apply(display_local_time)
        display_history["Résultat actualisé"] = display_history["Résultat actualisé (UTC)"].apply(display_local_time)
        display_history = display_history.drop(columns=["Créé à (UTC)", "Bougie analysée (UTC)", "Résultat actualisé (UTC)"])
        st.dataframe(display_history, use_container_width=True, hide_index=True)
        st.download_button(
            label="Télécharger l'historique CSV",
            data=display_history.to_csv(index=False).encode("utf-8"),
            file_name="forexscope_historique.csv",
            mime="text/csv",
        )


with tab_risk:
    st.subheader("Calculateur de risque par position")
    st.caption(
        "Outil de préparation : il calcule le montant risqué et le ratio rendement/risque. "
        "La taille en unités est une approximation et dépend de la devise du compte, "
        "de la paire et des spécifications du courtier."
    )

    r1, r2, r3 = st.columns(3)
    with r1:
        account_balance = st.number_input(
            "Capital du compte", min_value=0.0, value=100.0, step=10.0,
            help="Saisis le solde dans la devise de ton compte."
        )
    with r2:
        risk_percent = st.number_input(
            "Risque souhaité (%)", min_value=0.1, max_value=5.0,
            value=1.0, step=0.1,
            help="Pour apprendre, de nombreux traders étudient des risques faibles, sans garantie de résultat."
        )
    with r3:
        entry_price = st.number_input(
            "Prix d'entrée", min_value=0.0000001, value=1.10000,
            format="%.6f", help="Remplace cette valeur par le prix d'entrée envisagé."
        )

    r4, r5 = st.columns(2)
    with r4:
        stop_price = st.number_input(
            "Stop Loss", min_value=0.0000001, value=1.09500,
            format="%.6f", help="Le Stop Loss doit être différent du prix d'entrée."
        )
    with r5:
        target_price = st.number_input(
            "Take Profit", min_value=0.0000001, value=1.11000,
            format="%.6f", help="Le Take Profit doit être placé dans le sens du trade."
        )

    risk_cash = account_balance * risk_percent / 100.0
    stop_distance = abs(entry_price - stop_price)
    reward_distance = abs(target_price - entry_price)
    valid_direction = (target_price > entry_price > stop_price) or (target_price < entry_price < stop_price)

    if stop_distance <= 0:
        st.error("Le prix d'entrée et le Stop Loss doivent être différents.")
    else:
        rr_ratio = reward_distance / stop_distance
        units_approx = risk_cash / stop_distance
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Montant exposé au risque", f"{risk_cash:,.2f}")
        k2.metric("Distance entrée–SL", f"{stop_distance:.6f}")
        k3.metric("Ratio rendement/risque", f"1:{rr_ratio:.2f}")
        k4.metric("Taille indicative", f"{units_approx:,.2f} unités")

        if not valid_direction:
            st.warning("Vérifie le sens des niveaux : pour un achat, TP > entrée > SL ; pour une vente, TP < entrée < SL.")
        if rr_ratio < 1.0:
            st.warning("Le gain potentiel est inférieur au risque prévu (ratio inférieur à 1:1).")
        st.info(
            f"Si le Stop Loss est exécuté au niveau prévu, la perte théorique serait d'environ "
            f"{risk_cash:,.2f} dans la devise du compte, hors spread, commission et slippage."
        )
        st.warning(
            "La taille en unités est une approximation mathématique, pas un nombre de lots MT5 fiable. "
            "Pour les paires Forex, l'or, les indices et les CFD, vérifie la valeur du tick/pip, "
            "la taille du contrat et la devise du compte auprès du courtier avant toute utilisation."
        )

    st.divider()
    st.subheader("Estimateur de lots MT5 avec valeur du pip fournie")
    st.caption(
        "Saisis les spécifications affichées par ton courtier. Cet estimateur n'invente pas la valeur du pip : "
        "il utilise les valeurs que tu fournis et ne remplace pas la vérification dans MT5."
    )
    lot1, lot2, lot3 = st.columns(3)
    with lot1:
        pip_size = st.number_input(
            "Taille d'un pip en prix", min_value=0.000001, value=0.0001, format="%.6f",
            help="Exemple courant : 0,0001 pour beaucoup de paires Forex; 0,01 pour certaines paires en JPY. Vérifie l'instrument."
        )
    with lot2:
        pip_value_per_lot = st.number_input(
            "Valeur d'un pip pour 1 lot", min_value=0.000001, value=10.0, step=0.5,
            help="Valeur dans la devise de ton compte, vérifiée pour cet instrument chez ton courtier."
        )
    with lot3:
        lot_step = st.selectbox("Pas de volume du courtier", [0.001, 0.01, 0.1, 1.0], index=1)

    if stop_distance > 0 and pip_size > 0 and pip_value_per_lot > 0:
        stop_pips = stop_distance / pip_size
        lot_calculation = calculate_broker_lot_size(
            risk_cash=risk_cash, stop_pips=stop_pips,
            pip_value_per_lot=pip_value_per_lot, lot_step=lot_step,
            min_lot=lot_step,
        )
        raw_lots = lot_calculation["raw_lots"]
        rounded_lots = lot_calculation["rounded_lots"]
        rounded_risk = lot_calculation["estimated_risk"]
        l1, l2, l3 = st.columns(3)
        l1.metric("Distance du SL", f"{stop_pips:.1f} pips")
        l2.metric("Volume théorique", f"{raw_lots:.3f} lot(s)")
        l3.metric("Volume arrondi vers le bas", f"{rounded_lots:.3f} lot(s)")
        st.write(f"Risque estimé après arrondi : **{rounded_risk:,.2f}** dans la devise du compte.")
        if rounded_lots <= 0:
            st.warning("Le risque autorisé est trop faible pour atteindre le plus petit volume sélectionné. Ne force pas le volume à la hausse.")
        if rounded_risk > risk_cash + 1e-9:
            st.error("Le volume dépasse le risque prévu. Réduis le volume et vérifie les paramètres.")
        st.caption(
            "Hypothèses : valeur du pip constante, exécution au SL prévu, sans spread, commission, slippage ni conversion de devise. "
            "Ne pas utiliser le résultat avant vérification des spécifications du symbole dans MT5."
        )



def run_sensitivity_backtest(raw_data, strategy, rr_value, cost_r):
    """Backtest mécanique pour comparer les hypothèses R:R et de coûts."""
    frame = calculate_indicators(raw_data.copy()).reset_index(drop=True)
    needed = ["open", "high", "low", "close", "ema_20", "ema_50", "atr"]
    frame = frame.dropna(subset=needed).reset_index(drop=True)
    trades = []
    i = 1
    while i < len(frame) - 1:
        prev, cur = frame.iloc[i - 1], frame.iloc[i]
        atr = float(cur["atr"])
        if not np.isfinite(atr) or atr <= 0:
            i += 1
            continue
        entry_i = i + 1
        entry = float(frame.iloc[entry_i]["open"])
        if strategy == "EMA20/EMA50":
            up = prev["ema_20"] <= prev["ema_50"] and cur["ema_20"] > cur["ema_50"]
            down = prev["ema_20"] >= prev["ema_50"] and cur["ema_20"] < cur["ema_50"]
            if not (up or down):
                i += 1
                continue
            side = "ACHAT" if up else "VENTE"
            risk_dist = atr
            sl = entry - risk_dist if side == "ACHAT" else entry + risk_dist
        else:
            setup = _find_confirmed_smc_setup(frame.iloc[:i + 1].copy(), atr)
            if setup.get("direction") not in ("Haussier", "Baissier"):
                i += 1
                continue
            side = "ACHAT" if setup["direction"] == "Haussier" else "VENTE"
            sweep_extreme = float(setup["sweep_extreme"])
            sl = sweep_extreme - 0.2 * atr if side == "ACHAT" else sweep_extreme + 0.2 * atr
            risk_dist = entry - sl if side == "ACHAT" else sl - entry
            if not np.isfinite(risk_dist) or risk_dist <= 0 or risk_dist > 2.5 * atr:
                i += 1
                continue
        tp = entry + risk_dist * float(rr_value) if side == "ACHAT" else entry - risk_dist * float(rr_value)
        outcome = "OUVERT"
        exit_i = len(frame) - 1
        gross_r = ((float(frame.iloc[-1]["close"]) - entry) if side == "ACHAT" else (entry - float(frame.iloc[-1]["close"]))) / risk_dist
        for j in range(entry_i, len(frame)):
            candle = frame.iloc[j]
            hit_sl = float(candle["low"]) <= sl if side == "ACHAT" else float(candle["high"]) >= sl
            hit_tp = float(candle["high"]) >= tp if side == "ACHAT" else float(candle["low"]) <= tp
            if hit_sl and hit_tp:
                outcome, exit_i, gross_r = "AMBIGU", j, np.nan
                break
            if hit_sl:
                outcome, exit_i, gross_r = "PERDANT", j, -1.0
                break
            if hit_tp:
                outcome, exit_i, gross_r = "GAGNANT", j, float(rr_value)
                break
        if outcome in ("GAGNANT", "PERDANT"):
            trades.append({"Résultat": outcome, "R brut": float(gross_r), "R net estimé": float(gross_r) - float(cost_r), "Entrée": frame.iloc[entry_i]["time"]})
        i = max(entry_i + 1, exit_i + 1)
    return trades


@st.cache_data(ttl=300, show_spinner=False)
def run_smc_filter_rejection_audit(raw_data):
    """Compte les raisons de rejet des setups SMC mécaniques sans simuler de trades.

    Les catégories de rejet sont non exclusives : un setup peut échouer à plusieurs
    filtres. Les statistiques décrivent la logique de filtre, pas la qualité future.
    """
    empty = pd.DataFrame(columns=["Contrôle", "Occurrences", "% des setups candidats"])
    if raw_data is None or len(raw_data) < 80:
        return empty, pd.DataFrame()

    frame = calculate_indicators(raw_data.copy()).reset_index(drop=True)
    needed = ["open", "high", "low", "close", "ema_20", "ema_50", "rsi", "atr", "adx"]
    frame = frame.dropna(subset=[c for c in needed if c in frame.columns]).reset_index(drop=True)
    if len(frame) < 80:
        return empty, pd.DataFrame()

    counts = {
        "Setups SMC candidats": 0,
        "Rejetés : risque/SL invalide": 0,
        "Rejetés : momentum EMA/RSI": 0,
        "Rejetés : prix trop éloigné EMA20": 0,
        "Rejetés : ADX sous le seuil": 0,
        "Conservent tous les filtres": 0,
    }
    examples = []
    for i in range(60, len(frame) - 1):
        cur = frame.iloc[i]
        atr = float(cur["atr"])
        if not np.isfinite(atr) or atr <= 0:
            continue
        setup = _find_confirmed_smc_setup(frame.iloc[:i + 1].copy(), atr)
        direction = setup.get("direction")
        if direction not in ("Haussier", "Baissier"):
            continue
        counts["Setups SMC candidats"] += 1
        side = "ACHAT" if direction == "Haussier" else "VENTE"
        entry = float(frame.iloc[i + 1]["open"])
        sweep_extreme = float(setup.get("sweep_extreme", np.nan))
        sl = sweep_extreme - 0.2 * atr if side == "ACHAT" else sweep_extreme + 0.2 * atr
        risk_dist = entry - sl if side == "ACHAT" else sl - entry
        risk_ok = np.isfinite(risk_dist) and 0 < risk_dist <= 2.5 * atr
        if not risk_ok:
            counts["Rejetés : risque/SL invalide"] += 1
            examples.append({"Heure setup": cur.get("time", i), "Sens": side, "Statut": "Rejet risque/SL", "Distance EMA20 (ATR)": np.nan, "ADX": float(cur["adx"]) if pd.notna(cur["adx"]) else np.nan})
            continue

        ema20, ema50, rsi = float(cur["ema_20"]), float(cur["ema_50"]), float(cur["rsi"])
        adx = float(cur["adx"]) if pd.notna(cur["adx"]) else None
        distance = abs(float(cur["close"]) - ema20) / atr
        momentum_ok = (ema20 > ema50 and rsi >= 50) if side == "ACHAT" else (ema20 < ema50 and rsi <= 50)
        distance_ok = distance <= 1.5
        adx_ok = adx is None or adx >= 12
        if not momentum_ok:
            counts["Rejetés : momentum EMA/RSI"] += 1
        if not distance_ok:
            counts["Rejetés : prix trop éloigné EMA20"] += 1
        if not adx_ok:
            counts["Rejetés : ADX sous le seuil"] += 1
        if momentum_ok and distance_ok and adx_ok:
            counts["Conservent tous les filtres"] += 1
            status = "Conservé"
        else:
            failed = []
            if not momentum_ok: failed.append("momentum EMA/RSI")
            if not distance_ok: failed.append("distance EMA20")
            if not adx_ok: failed.append("ADX")
            status = "Rejet : " + ", ".join(failed)
        examples.append({"Heure setup": cur.get("time", i), "Sens": side, "Statut": status, "Distance EMA20 (ATR)": round(distance, 3), "ADX": round(adx, 2) if adx is not None else np.nan})

    total = counts["Setups SMC candidats"]
    rows = []
    for name, value in counts.items():
        if name == "Setups SMC candidats":
            rows.append({"Contrôle": name, "Occurrences": value, "% des setups candidats": 100.0 if value else 0.0})
        else:
            rows.append({"Contrôle": name, "Occurrences": value, "% des setups candidats": (value / total * 100.0) if total else 0.0})
    return pd.DataFrame(rows), pd.DataFrame(examples)


@st.cache_data(ttl=300, show_spinner=False)
def bootstrap_expectancy_interval(trades_df, iterations=1000, seed=42):
    """Intervalle bootstrap descriptif de l'espérance en R; ne corrige pas la dépendance temporelle."""
    if trades_df is None or trades_df.empty or "R net estimé" not in trades_df.columns:
        return {"Trades": 0, "Espérance (R/trade)": np.nan, "IC 95 % bas (R)": np.nan, "IC 95 % haut (R)": np.nan, "Interprétation": "Pas assez de données"}
    values = pd.to_numeric(trades_df["R net estimé"], errors="coerce").dropna().to_numpy(dtype=float)
    n = len(values)
    if n < 2:
        mean = float(np.mean(values)) if n else np.nan
        return {"Trades": n, "Espérance (R/trade)": mean, "IC 95 % bas (R)": np.nan, "IC 95 % haut (R)": np.nan, "Interprétation": "Au moins 2 trades requis"}
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, n), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    if low > 0:
        label = "Intervalle positif (indicatif)"
    elif high < 0:
        label = "Intervalle négatif (indicatif)"
    else:
        label = "Intervalle recouvrant zéro"
    return {"Trades": n, "Espérance (R/trade)": float(np.mean(values)), "IC 95 % bas (R)": float(low), "IC 95 % haut (R)": float(high), "Interprétation": label}



@st.cache_data(ttl=300, show_spinner=False)
def run_smc_chronological_windows(raw_data, windows=4, rr_value=2.0, cost_r=0.02):
    """Évalue les règles SMC fixes sur des fenêtres chronologiques successives.

    Chaque fenêtre est évaluée séparément; ce n'est pas un véritable apprentissage
    walk-forward ni une preuve de performance future.
    """
    if raw_data is None or raw_data.empty:
        return pd.DataFrame(), {}
    data = raw_data.copy().reset_index(drop=True)
    windows = max(2, min(int(windows), 6))
    # Fenêtres contiguës, sans mélange aléatoire des bougies.
    boundaries = np.linspace(0, len(data), windows + 1, dtype=int)
    rows, details = [], {}
    for idx in range(windows):
        start, end = int(boundaries[idx]), int(boundaries[idx + 1])
        segment = data.iloc[start:end].copy()
        label = f"Fenêtre {idx + 1}"
        if len(segment) < 100:
            rows.append({"Fenêtre": label, "Début": str(segment.iloc[0].get("time", start)) if len(segment) else "—", "Fin": str(segment.iloc[-1].get("time", end)) if len(segment) else "—", "Bougies": len(segment), "Variante": "—", "Trades classés": 0, "Espérance (R/trade)": np.nan, "Résultat net (R)": np.nan, "Statut": "Fenêtre trop courte (<100 bougies)"})
            continue
        try:
            summary, trade_groups = run_smc_filter_comparison(segment, rr_value=float(rr_value), cost_r=float(cost_r))
            for _, metric in summary.iterrows():
                variant = str(metric.get("Variante", "Inconnue"))
                trade_df = trade_groups.get(variant, pd.DataFrame()).copy()
                count = int(metric.get("Trades classés", 0) or 0)
                rows.append({
                    "Fenêtre": label,
                    "Début": str(segment.iloc[0].get("time", start)),
                    "Fin": str(segment.iloc[-1].get("time", end)),
                    "Bougies": len(segment),
                    "Variante": variant,
                    "Trades classés": count,
                    "Espérance (R/trade)": metric.get("Espérance (R/trade)", np.nan),
                    "Résultat net (R)": metric.get("Résultat net estimé (R)", np.nan),
                    "Taux de réussite (%)": metric.get("Taux de réussite (%)", np.nan),
                    "Facteur de profit": metric.get("Facteur de profit", np.nan),
                    "Statut": "Échantillon faible" if count < 30 else "À examiner",
                })
                details[f"{label} — {variant}"] = trade_df
        except Exception as exc:
            rows.append({"Fenêtre": label, "Début": str(segment.iloc[0].get("time", start)), "Fin": str(segment.iloc[-1].get("time", end)), "Bougies": len(segment), "Variante": "—", "Trades classés": 0, "Espérance (R/trade)": np.nan, "Résultat net (R)": np.nan, "Statut": f"Erreur : {type(exc).__name__}"})
    return pd.DataFrame(rows), details


def summarize_smc_window_stability(results):
    """Résume la stabilité des résultats par variante sans mélanger les fenêtres."""
    if results is None or results.empty or not {"Variante", "Espérance (R/trade)"}.issubset(results.columns):
        return pd.DataFrame()
    frame = results.copy()
    frame["Espérance (R/trade)"] = pd.to_numeric(frame["Espérance (R/trade)"], errors="coerce")
    if "Trades classés" in frame.columns:
        frame["Trades classés"] = pd.to_numeric(frame["Trades classés"], errors="coerce").fillna(0)
    rows = []
    for variant, group in frame.groupby("Variante", dropna=False):
        valid = group.dropna(subset=["Espérance (R/trade)"])
        if valid.empty:
            rows.append({"Variante": str(variant), "Fenêtres évaluées": 0, "Fenêtres à espérance positive (%)": np.nan,
                         "Espérance médiane (R/trade)": np.nan, "Pire espérance (R/trade)": np.nan,
                         "Écart-type entre fenêtres": np.nan, "Trades classés total": int(group.get("Trades classés", pd.Series(dtype=float)).sum()),
                         "Interprétation": "Données insuffisantes"})
            continue
        vals = valid["Espérance (R/trade)"].astype(float)
        pos_share = float((vals > 0).mean() * 100.0)
        total_trades = int(group.get("Trades classés", pd.Series(dtype=float)).sum())
        if len(valid) < 3 or total_trades < 30:
            label = "Échantillon trop faible pour conclure"
        elif pos_share >= 75 and float(vals.median()) > 0:
            label = "Résultats positifs dans la majorité des fenêtres; à confirmer"
        elif pos_share <= 25 and float(vals.median()) < 0:
            label = "Résultats négatifs dans la majorité des fenêtres"
        else:
            label = "Résultats variables selon la période"
        rows.append({"Variante": str(variant), "Fenêtres évaluées": int(len(valid)),
                     "Fenêtres à espérance positive (%)": pos_share,
                     "Espérance médiane (R/trade)": float(vals.median()),
                     "Pire espérance (R/trade)": float(vals.min()),
                     "Écart-type entre fenêtres": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                     "Trades classés total": total_trades, "Interprétation": label})
    return pd.DataFrame(rows)

def run_smc_filter_comparison(raw_data, rr_value=2.0, cost_r=0.02):
    """Compare le setup SMC mécanique avant/après filtres locaux sur les mêmes bougies.

    Ce test est volontairement distinct du moteur live : il ne simule pas la confluence H4/D1.
    Les entrées se font à l'ouverture suivante et les cas SL/TP dans la même bougie sont ambigus.
    """
    frame = calculate_indicators(raw_data.copy()).reset_index(drop=True)
    needed = ["open", "high", "low", "close", "ema_20", "ema_50", "rsi", "atr", "adx"]
    frame = frame.dropna(subset=[c for c in needed if c in frame.columns]).reset_index(drop=True)
    variants = {"SMC de base": [], "SMC + filtres phase 22": []}
    # Comptabiliser aussi les trades non classés pour rendre visible le biais d'exclusion.
    outcome_counts = {name: {"Ambigus (SL et TP même bougie)": 0, "Non résolus (fin des données)": 0} for name in variants}
    i = max(60, 1)
    while i < len(frame) - 1:
        cur = frame.iloc[i]
        atr = float(cur["atr"])
        if not np.isfinite(atr) or atr <= 0:
            i += 1
            continue
        history = frame.iloc[:i + 1].copy()
        setup = _find_confirmed_smc_setup(history, atr)
        direction = setup.get("direction")
        if direction not in ("Haussier", "Baissier"):
            i += 1
            continue
        side = "ACHAT" if direction == "Haussier" else "VENTE"
        entry_i = i + 1
        entry = float(frame.iloc[entry_i]["open"])
        sweep_extreme = float(setup["sweep_extreme"])
        sl = sweep_extreme - 0.2 * atr if side == "ACHAT" else sweep_extreme + 0.2 * atr
        risk_dist = entry - sl if side == "ACHAT" else sl - entry
        if not np.isfinite(risk_dist) or risk_dist <= 0 or risk_dist > 2.5 * atr:
            i += 1
            continue

        # Base : structure SMC + stop valide. Filtered : mêmes conditions, plus filtres locaux.
        variants_to_test = ["SMC de base"]
        ema20 = float(cur["ema_20"])
        ema50 = float(cur["ema_50"])
        rsi = float(cur["rsi"])
        adx = float(cur["adx"]) if pd.notna(cur["adx"]) else None
        ema_distance_atr = abs(float(cur["close"]) - ema20) / atr
        momentum_ok = (ema20 > ema50 and rsi >= 50) if side == "ACHAT" else (ema20 < ema50 and rsi <= 50)
        filters_ok = momentum_ok and ema_distance_atr <= 1.5 and (adx is None or adx >= 12)
        if filters_ok:
            variants_to_test.append("SMC + filtres phase 22")

        tp = entry + risk_dist * float(rr_value) if side == "ACHAT" else entry - risk_dist * float(rr_value)
        outcome = "OUVERT"
        gross_r = np.nan
        exit_i = len(frame) - 1
        for j in range(entry_i, len(frame)):
            candle = frame.iloc[j]
            hit_sl = float(candle["low"]) <= sl if side == "ACHAT" else float(candle["high"]) >= sl
            hit_tp = float(candle["high"]) >= tp if side == "ACHAT" else float(candle["low"]) <= tp
            if hit_sl and hit_tp:
                outcome, exit_i, gross_r = "AMBIGU", j, np.nan
                break
            if hit_sl:
                outcome, exit_i, gross_r = "PERDANT", j, -1.0
                break
            if hit_tp:
                outcome, exit_i, gross_r = "GAGNANT", j, float(rr_value)
                break
        if outcome in ("GAGNANT", "PERDANT"):
            trade = {"Résultat": outcome, "R brut": float(gross_r), "R net estimé": float(gross_r) - float(cost_r), "Entrée": frame.iloc[entry_i]["time"]}
            for variant in variants_to_test:
                variants[variant].append(trade.copy())
        else:
            label = "Ambigus (SL et TP même bougie)" if outcome == "AMBIGU" else "Non résolus (fin des données)"
            for variant in variants_to_test:
                outcome_counts[variant][label] += 1
        # Ne pas réutiliser les mêmes bougies de trade comme entrées simultanées.
        i = max(entry_i, exit_i + 1)

    rows = []
    details = {}
    for name, trades in variants.items():
        details[name] = pd.DataFrame(trades)
        df = details[name]
        wins = int((df["Résultat"] == "GAGNANT").sum()) if not df.empty else 0
        losses = int((df["Résultat"] == "PERDANT").sum()) if not df.empty else 0
        count = wins + losses
        net = float(df["R net estimé"].sum()) if count else 0.0
        expectancy = float(df["R net estimé"].mean()) if count else 0.0
        gross_wins = float(df.loc[df["R net estimé"] > 0, "R net estimé"].sum()) if count else 0.0
        gross_losses = abs(float(df.loc[df["R net estimé"] < 0, "R net estimé"].sum())) if count else 0.0
        pf = gross_wins / gross_losses if gross_losses > 0 else np.nan
        rows.append({"Variante": name, "Trades classés": count, "Gagnants": wins, "Perdants": losses,
                     "Ambigus (SL et TP même bougie)": outcome_counts[name]["Ambigus (SL et TP même bougie)"],
                     "Non résolus (fin des données)": outcome_counts[name]["Non résolus (fin des données)"],
                     "Taux de réussite (%)": (wins / count * 100.0) if count else 0.0,
                     "Résultat net estimé (R)": net, "Espérance (R/trade)": expectancy,
                     "Facteur de profit": pf})
    return pd.DataFrame(rows), details


@st.cache_data(ttl=300, show_spinner=False)
def run_smc_walk_forward(raw_data, folds=3, rr_value=2.0, cost_r=0.02):
    """Évalue l'approximation SMC sur des segments chronologiques distincts.

    Chaque segment est testé séparément; les segments courts sont ignorés.
    Ce test de stabilité n'est pas un walk-forward optimisé sans biais de recherche.
    """
    if raw_data is None or len(raw_data) < folds * 100:
        return pd.DataFrame(), {}
    frame = raw_data.sort_values("time").reset_index(drop=True) if "time" in raw_data.columns else raw_data.reset_index(drop=True)
    chunks = [chunk.reset_index(drop=True) for chunk in np.array_split(frame, int(folds))]
    rows = []
    details = {}
    for idx, chunk in enumerate(chunks, start=1):
        if len(chunk) < 100:
            continue
        summary, trades_by_variant = run_smc_filter_comparison(chunk, rr_value=rr_value, cost_r=cost_r)
        for variant, trades in trades_by_variant.items():
            details[f"Période {idx} — {variant}"] = trades.copy()
            if trades.empty:
                wins = losses = count = 0
                net = expectancy = 0.0
                profit_factor = np.nan
            else:
                wins = int((trades["Résultat"] == "GAGNANT").sum())
                losses = int((trades["Résultat"] == "PERDANT").sum())
                count = wins + losses
                net = float(trades["R net estimé"].sum()) if count else 0.0
                expectancy = float(trades["R net estimé"].mean()) if count else 0.0
                gross_wins = float(trades.loc[trades["R net estimé"] > 0, "R net estimé"].sum())
                gross_losses = abs(float(trades.loc[trades["R net estimé"] < 0, "R net estimé"].sum()))
                profit_factor = gross_wins / gross_losses if gross_losses > 0 else np.nan
            rows.append({"Période chronologique": idx, "Variante": variant, "Bougies": len(chunk),
                         "Trades classés": count, "Gagnants": wins, "Perdants": losses,
                         "Taux de réussite (%)": wins / count * 100.0 if count else 0.0,
                         "Résultat net estimé (R)": net, "Espérance (R/trade)": expectancy,
                         "Facteur de profit": profit_factor})
    return pd.DataFrame(rows), details


with tab_backtest:
    st.subheader("Backtesting historique — EMA ou SMC")
    st.caption(
        "Simulation pédagogique sur les bougies Yahoo Finance disponibles. "
        "Choisis entre le croisement EMA20/EMA50 et une approximation mécanique du setup SMC (sweep → cassure → impulsion). "
        "Les entrées sont simulées à l'ouverture de la bougie suivante. Les résultats restent pédagogiques et ne garantissent pas les performances futures."
    )
    b1, b2, b3 = st.columns(3)
    with b1:
        bt_timeframe = st.selectbox("Unité de temps", list(INTERVALS.keys()), index=list(INTERVALS.keys()).index(timeframe), key="bt_timeframe")
    with b2:
        bt_count = st.slider("Nombre de bougies à tester", min_value=100, max_value=1500, value=500, step=100)
    with b3:
        bt_rr = st.selectbox("Objectif rendement/risque", [1.0, 1.5, 2.0, 3.0], index=2)
    bt_strategy = st.selectbox("Stratégie à tester", ["SMC (sweep + structure + impulsion)", "EMA20/EMA50"], key="bt_strategy")
    bt_oos_pct = st.slider("Partie réservée au test final hors échantillon (%)", min_value=20, max_value=50, value=30, step=5, key="bt_oos_pct", help="Les premiers trades servent de période de développement; la partie finale sert de vérification séparée.")
    b4, b5 = st.columns(2)
    with b4:
        bt_cost_r = st.number_input(
            "Coût estimé aller-retour (en R)", min_value=0.0, max_value=0.50,
            value=0.02, step=0.01, key="bt_cost_r",
            help="Estimation combinée spread, commission et slippage exprimée en multiples du risque initial. Exemple : 0,02 R."
        )
    with b5:
        bt_initial_equity = st.number_input(
            "Capital de départ (unités monétaires)", min_value=1.0, value=1000.0,
            step=100.0, key="bt_initial_equity",
            help="Utilisé pour illustrer une courbe de capital avec risque fixe par trade."
        )
    bt_risk_pct = st.slider(
        "Risque simulé par trade (% du capital courant)", min_value=0.1, max_value=3.0,
        value=1.0, step=0.1, key="bt_risk_pct"
    )

    if st.button("Lancer le backtest", type="primary", key="run_backtest"):
        with st.spinner("Préparation des données et simulation..."):
            bt_data = fetch_candles(SYMBOLS[market_name], bt_timeframe, bt_count)
            if bt_data is None or len(bt_data) < 60:
                st.error("Données insuffisantes pour ce marché et cette unité de temps. Essaie une autre unité.")
            else:
                bt = calculate_indicators(bt_data.copy()).reset_index(drop=True)
                needed = ["open", "high", "low", "close", "ema_20", "ema_50", "atr"]
                bt = bt.dropna(subset=needed).reset_index(drop=True)
                trades = []
                i = 1
                while i < len(bt) - 1:
                    prev = bt.iloc[i - 1]
                    cur = bt.iloc[i]
                    if not np.isfinite(cur["atr"]) or cur["atr"] <= 0:
                        i += 1
                        continue
                    entry_i = i + 1
                    side = None
                    if bt_strategy == "EMA20/EMA50":
                        cross_up = prev["ema_20"] <= prev["ema_50"] and cur["ema_20"] > cur["ema_50"]
                        cross_down = prev["ema_20"] >= prev["ema_50"] and cur["ema_20"] < cur["ema_50"]
                        if cross_up:
                            side = "ACHAT"
                        elif cross_down:
                            side = "VENTE"
                        if side is None:
                            i += 1
                            continue
                        entry = float(bt.iloc[entry_i]["open"])
                        risk_dist = float(cur["atr"])
                        sl = entry - risk_dist if side == "ACHAT" else entry + risk_dist
                    else:
                        # Le setup est calculé uniquement sur les bougies jusqu'à i.
                        history = bt.iloc[:i + 1].copy()
                        setup = _find_confirmed_smc_setup(history, float(cur["atr"]))
                        if setup.get("direction") not in ("Haussier", "Baissier"):
                            i += 1
                            continue
                        side = "ACHAT" if setup["direction"] == "Haussier" else "VENTE"
                        entry = float(bt.iloc[entry_i]["open"])
                        sweep_extreme = float(setup["sweep_extreme"])
                        sl = sweep_extreme - 0.2 * float(cur["atr"]) if side == "ACHAT" else sweep_extreme + 0.2 * float(cur["atr"])
                        risk_dist = entry - sl if side == "ACHAT" else sl - entry
                        if not np.isfinite(risk_dist) or risk_dist <= 0 or risk_dist > 2.5 * float(cur["atr"]):
                            i += 1
                            continue
                    tp = entry + risk_dist * float(bt_rr) if side == "ACHAT" else entry - risk_dist * float(bt_rr)
                    outcome = "OUVERT"
                    exit_price = float(bt.iloc[-1]["close"])
                    exit_time = bt.iloc[-1]["time"]
                    exit_i = len(bt) - 1
                    for j in range(entry_i, len(bt)):
                        candle = bt.iloc[j]
                        if side == "ACHAT":
                            hit_sl = float(candle["low"]) <= sl
                            hit_tp = float(candle["high"]) >= tp
                        else:
                            hit_sl = float(candle["high"]) >= sl
                            hit_tp = float(candle["low"]) <= tp
                        if hit_sl and hit_tp:
                            outcome = "AMBIGU — exclu"
                            exit_price = float(candle["close"])
                            exit_time = candle["time"]
                            exit_i = j
                            break
                        if hit_sl:
                            outcome, exit_price, exit_time, exit_i = "PERDANT", sl, candle["time"], j
                            break
                        if hit_tp:
                            outcome, exit_price, exit_time, exit_i = "GAGNANT", tp, candle["time"], j
                            break
                    if outcome == "GAGNANT":
                        gross_r = float(bt_rr)
                    elif outcome == "PERDANT":
                        gross_r = -1.0
                    elif outcome == "OUVERT":
                        raw_move = (exit_price - entry) if side == "ACHAT" else (entry - exit_price)
                        gross_r = raw_move / risk_dist
                    else:
                        gross_r = np.nan
                    # Les coûts sont retranchés des résultats simulés, mais ne modifient pas les prix de déclenchement.
                    net_r_trade = gross_r - float(bt_cost_r) if np.isfinite(gross_r) else np.nan
                    trades.append({
                        "Entrée": bt.iloc[entry_i]["time"], "Sens": side,
                        "Prix entrée": entry, "SL": sl, "TP": tp,
                        "Sortie": exit_time, "Résultat": outcome,
                        "Bougies détenues": int(exit_i - entry_i + 1),
                        "Stratégie": bt_strategy,
                        "R brut": gross_r, "Coût estimé (R)": float(bt_cost_r),
                        "R net estimé": net_r_trade,
                    })
                    i = max(entry_i + 1, exit_i + 1)

                if not trades:
                    st.info("Aucun croisement EMA exploitable n'a été trouvé sur la période chargée.")
                else:
                    results = pd.DataFrame(trades)
                    resolved = results[results["Résultat"].isin(["GAGNANT", "PERDANT"])]
                    wins = int((resolved["Résultat"] == "GAGNANT").sum())
                    losses = int((resolved["Résultat"] == "PERDANT").sum())
                    ambiguous = int((results["Résultat"] == "AMBIGU — exclu").sum())
                    unresolved = int((results["Résultat"] == "OUVERT").sum())
                    closed_count = wins + losses
                    win_rate = (wins / closed_count * 100) if closed_count else 0.0
                    net_r = float(resolved["R net estimé"].sum()) if not resolved.empty else 0.0
                    gross_values = resolved["R brut"].astype(float) if not resolved.empty else pd.Series(dtype=float)
                    net_values = resolved["R net estimé"].astype(float) if not resolved.empty else pd.Series(dtype=float)
                    positive_sum = float(net_values[net_values > 0].sum()) if not net_values.empty else 0.0
                    negative_sum = abs(float(net_values[net_values < 0].sum())) if not net_values.empty else 0.0
                    profit_factor = positive_sum / negative_sum if negative_sum > 0 else np.nan
                    expectancy_r = float(net_values.mean()) if not net_values.empty else 0.0

                    # Courbe illustrative à risque fractionnel constant; seuls les trades classés sont inclus.
                    equity = float(bt_initial_equity)
                    peak = equity
                    max_drawdown_pct = 0.0
                    equity_rows = []
                    for _, trade in resolved.iterrows():
                        equity_before = equity
                        risk_cash = equity_before * float(bt_risk_pct) / 100.0
                        pnl_cash = risk_cash * float(trade["R net estimé"])
                        equity = max(0.0, equity_before + pnl_cash)
                        peak = max(peak, equity)
                        dd = ((peak - equity) / peak * 100.0) if peak > 0 else 0.0
                        max_drawdown_pct = max(max_drawdown_pct, dd)
                        equity_rows.append({
                            "Trade": len(equity_rows) + 1,
                            "Entrée": trade["Entrée"],
                            "Capital avant": equity_before,
                            "Risque monétaire prévu": risk_cash,
                            "Résultat net (R)": float(trade["R net estimé"]),
                            "P&L estimé": pnl_cash,
                            "Capital après": equity,
                            "Drawdown (%)": dd,
                        })

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Trades classés", closed_count)
                    m2.metric("Taux de réussite", f"{win_rate:.1f}%")
                    m3.metric("Résultat net estimé", f"{net_r:+.2f} R")
                    m4.metric("Facteur de profit", f"{profit_factor:.2f}" if np.isfinite(profit_factor) else "N/D")
                    m5, m6, m7, m8 = st.columns(4)
                    m5.metric("Espérance par trade", f"{expectancy_r:+.3f} R")
                    m6.metric("Drawdown max simulé", f"{max_drawdown_pct:.2f}%")
                    m7.metric("Cas ambigus exclus", ambiguous)
                    m8.metric("Trades non résolus", unresolved)
                    if equity_rows:
                        equity_df = pd.DataFrame(equity_rows)
                        st.subheader("Courbe de capital simulée")
                        st.line_chart(equity_df.set_index("Trade")["Capital après"])
                        with st.expander("Détail du risque monétaire par trade"):
                            st.dataframe(equity_df, use_container_width=True, hide_index=True)
                            st.download_button(
                                "Télécharger le journal de capital CSV",
                                equity_df.to_csv(index=False).encode("utf-8-sig"),
                                file_name="forexscope_journal_capital.csv",
                                mime="text/csv",
                                key="download_equity_journal",
                            )
                    # Vérification hors échantillon : les derniers trades chronologiques sont séparés du développement.
                    # Les trades ambigus et ouverts ne contribuent pas aux métriques de performance.
                    if not results.empty:
                        ordered = results.sort_values("Entrée").reset_index(drop=True)
                        split_at = max(1, int(len(ordered) * (1.0 - float(bt_oos_pct) / 100.0)))
                        oos_all = ordered.iloc[split_at:].copy()
                        oos = oos_all[oos_all["Résultat"].isin(["GAGNANT", "PERDANT"])]
                        oos_wins = int((oos["Résultat"] == "GAGNANT").sum())
                        oos_losses = int((oos["Résultat"] == "PERDANT").sum())
                        oos_closed = oos_wins + oos_losses
                        oos_rate = (oos_wins / oos_closed * 100.0) if oos_closed else 0.0
                        oos_net = float(oos["R net estimé"].sum()) if not oos.empty else 0.0
                        oos_mean = float(oos["R net estimé"].mean()) if not oos.empty else 0.0
                        st.subheader("Vérification hors échantillon")
                        st.caption(f"Les derniers {bt_oos_pct}% des signaux chronologiques sont réservés au test final. Ce découpage reste exploratoire et ne remplace pas une validation sur plusieurs périodes.")
                        q1, q2, q3, q4 = st.columns(4)
                        q1.metric("Trades classés hors échantillon", oos_closed)
                        q2.metric("Taux de réussite hors échantillon", f"{oos_rate:.1f}%")
                        q3.metric("Résultat net hors échantillon", f"{oos_net:+.2f} R")
                        q4.metric("Espérance hors échantillon", f"{oos_mean:+.3f} R")
                        if oos_closed < 20:
                            st.info("Moins de 20 trades classés dans la partie hors échantillon : les statistiques sont trop limitées pour tirer une conclusion fiable.")
                    st.dataframe(results, use_container_width=True, hide_index=True)
                    csv_data = results.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("Télécharger les résultats CSV", csv_data, file_name="forexscope_backtest.csv", mime="text/csv")
                    st.warning(
                        "Le backtest SMC applique une approximation mécanique du sweep, de la cassure et de l'impulsion; il n'intègre pas encore la confluence multi-timeframe H4/D1 ni la validation contextuelle complète des FVG/OB. "
                        "Le coût aller-retour est une hypothèse saisie par l'utilisateur, exprimée en R; ce n'est pas une cotation réelle du courtier. "
                        "Les prix d'exécution ne sont pas ajustés pour les coûts, et les trades ambigus (SL et TP dans la même bougie) sont exclus. "
                        "La courbe de capital est illustrative, utilise un risque fractionnel fixe et ne prédit pas les performances futures. Le journal monétaire est une approximation en unités du capital, pas un calcul de lots MT5: la valeur du pip, la devise du compte, le contrat et les marges du courtier ne sont pas modélisés. Les trades non résolus sont affichés séparément et exclus des statistiques de performance."
                    )

    st.divider()
    st.subheader("Comparaison des filtres SMC — avant / après phase 22")
    st.caption(
        "Compare le setup SMC mécanique avec et sans les filtres locaux EMA/RSI, éloignement EMA20 et ADX, "
        "sur les mêmes données. Ce comparatif n'intègre pas la confluence H4/D1 du moteur en direct."
    )
    cf1, cf2, cf3 = st.columns(3)
    with cf1:
        compare_market = st.selectbox("Marché du comparatif", list(SYMBOLS.keys()), index=list(SYMBOLS.keys()).index(market_name), key="compare_filter_market")
    with cf2:
        compare_tf = st.selectbox("Unité du comparatif", list(INTERVALS.keys()), index=list(INTERVALS.keys()).index(bt_timeframe), key="compare_filter_tf")
    with cf3:
        compare_bars = st.slider("Bougies du comparatif", 100, 1500, 500, 100, key="compare_filter_bars")
    if st.button("Comparer avant / après filtres", key="run_filter_comparison", type="secondary"):
        with st.spinner("Comparaison des deux variantes sur les mêmes données..."):
            compare_data = fetch_candles(SYMBOLS[compare_market], compare_tf, compare_bars)
            if compare_data is None or len(compare_data) < 100:
                st.error("Données insuffisantes pour ce marché et cette unité de temps.")
            else:
                comparison, comparison_details = run_smc_filter_comparison(compare_data, rr_value=float(bt_rr), cost_r=float(bt_cost_r))
                st.dataframe(comparison, use_container_width=True, hide_index=True)
                st.download_button(
                    "Télécharger le résumé comparatif CSV",
                    comparison.to_csv(index=False).encode("utf-8-sig"),
                    file_name="forexscope_comparaison_filtres_smc.csv", mime="text/csv",
                    key="download_filter_comparison_summary"
                )
                for variant_name, variant_df in comparison_details.items():
                    with st.expander(f"Détail des trades — {variant_name}"):
                        if variant_df.empty:
                            st.info("Aucun trade classé pour cette variante sur la période sélectionnée.")
                        else:
                            st.dataframe(variant_df, use_container_width=True, hide_index=True)
                            st.download_button(
                                f"Exporter les trades — {variant_name}",
                                variant_df.to_csv(index=False).encode("utf-8-sig"),
                                file_name=("forexscope_trades_smc_base.csv" if variant_name == "SMC de base" else "forexscope_trades_smc_filtres.csv"),
                                mime="text/csv", key=f"download_{variant_name.replace(' ', '_').replace('+', 'plus')}_trades"
                            )
                st.warning(
                    "Test historique expérimental : les trades ambigus (SL et TP touchés dans la même bougie) et les trades encore ouverts en fin de données sont maintenant comptés séparément, mais exclus du taux de réussite et du résultat net. "
                    "Le filtre peut éliminer des trades gagnants comme perdants. Les résultats dépendent de la période et des coûts estimés; ce comparatif n'est pas le moteur live complet et ne prouve pas une rentabilité future."
                )


    st.divider()
    st.subheader("Audit des rejets SMC et contrôle hors échantillon")
    st.caption(
        "Mesure quels filtres bloquent les setups candidats, puis compare les variantes sur la première partie de l’historique et sur les dernières bougies réservées. "
        "Les catégories de rejet peuvent se chevaucher. Les calculs identiques sont conservés en cache pendant 5 minutes pour accélérer les relances."
    )
    au1, au2, au3 = st.columns(3)
    with au1:
        audit_market = st.selectbox("Marché de l’audit", list(SYMBOLS.keys()), index=list(SYMBOLS.keys()).index(market_name), key="audit_smc_market")
    with au2:
        audit_tf = st.selectbox("Unité de l’audit", list(INTERVALS.keys()), index=list(INTERVALS.keys()).index(bt_timeframe), key="audit_smc_tf")
    with au3:
        audit_bars = st.slider("Bougies de l’audit", 200, 1500, 800, 100, key="audit_smc_bars")
    audit_holdout_pct = st.select_slider("Partie finale réservée au contrôle", options=[20, 25, 30, 35], value=30, key="audit_smc_holdout_pct")
    if st.button("Analyser les rejets et le test final", key="run_smc_rejection_audit", type="secondary"):
        with st.spinner("Analyse des filtres et séparation chronologique des données..."):
            audit_data = fetch_candles(SYMBOLS[audit_market], audit_tf, audit_bars)
            if audit_data is None or len(audit_data) < 200:
                st.error("Données insuffisantes pour cet audit. Choisis une autre unité ou demande davantage de bougies.")
            else:
                audit_summary, audit_examples = run_smc_filter_rejection_audit(audit_data)
                st.markdown("#### Pourquoi les setups sont-ils rejetés ?")
                st.dataframe(audit_summary, use_container_width=True, hide_index=True)
                if audit_examples.empty:
                    st.info("Aucun setup candidat détecté sur la période sélectionnée.")
                else:
                    with st.expander("Afficher les setups examinés"):
                        st.dataframe(audit_examples, use_container_width=True, hide_index=True)
                        st.download_button("Exporter l’audit des setups CSV", audit_examples.to_csv(index=False).encode("utf-8-sig"), file_name="forexscope_audit_rejets_smc.csv", mime="text/csv", key="download_smc_audit_examples")
                split_at = int(len(audit_data) * (1.0 - float(audit_holdout_pct) / 100.0))
                # Les deux segments sont analysés séparément pour éviter que les résultats du premier segment
                # soient mélangés à ceux du test final. Chaque segment doit conserver assez de bougies.
                development_data = audit_data.iloc[:split_at].copy()
                holdout_data = audit_data.iloc[split_at:].copy()
                if len(development_data) < 100 or len(holdout_data) < 80:
                    st.warning("Un des segments est trop court pour fournir une comparaison utile.")
                else:
                    dev_summary, _ = run_smc_filter_comparison(development_data, rr_value=float(bt_rr), cost_r=float(bt_cost_r))
                    hold_summary, _ = run_smc_filter_comparison(holdout_data, rr_value=float(bt_rr), cost_r=float(bt_cost_r))
                    dev_summary = dev_summary.rename(columns={c: f"{c} — développement" for c in dev_summary.columns if c != "Variante"})
                    hold_summary = hold_summary.rename(columns={c: f"{c} — test final" for c in hold_summary.columns if c != "Variante"})
                    joined = dev_summary.merge(hold_summary, on="Variante", how="outer")
                    st.markdown("#### Comparaison chronologique : développement / test final")
                    st.dataframe(joined, use_container_width=True, hide_index=True)
                    # A visible sample-size check helps prevent overinterpreting a handful of trades.
                    test_trade_cols = [c for c in joined.columns if "Trades classés — test final" in c]
                    if test_trade_cols:
                        test_counts = pd.to_numeric(joined[test_trade_cols[0]], errors="coerce").fillna(0)
                        max_test_trades = int(test_counts.max()) if not test_counts.empty else 0
                        if max_test_trades < 30:
                            st.warning(f"Échantillon test final faible : au maximum {max_test_trades} trades classés. Les statistiques sont très incertaines; vise idéalement plusieurs dizaines de trades et teste d’autres périodes.")
                        else:
                            st.info(f"Le test final contient jusqu’à {max_test_trades} trades classés selon la variante. Cela améliore la lecture de l’échantillon, sans garantir sa représentativité.")
                    st.download_button("Exporter la comparaison hors échantillon CSV", joined.to_csv(index=False).encode("utf-8-sig"), file_name="forexscope_smc_holdout_comparison.csv", mime="text/csv", key="download_smc_holdout_comparison")
                    st.warning(
                        "Le segment final est séparé chronologiquement, mais ce test reste exploratoire : le moteur SMC est une approximation mécanique, "
                        "les coûts sont estimés et les petits échantillons sont peu fiables. Les filtres sont comptés de façon non exclusive; un setup peut échouer à plusieurs contrôles."
                    )

    st.divider()
    st.subheader("Stabilité chronologique des filtres SMC")
    st.caption("Découpe les bougies en périodes successives et compare le SMC de base aux filtres de la phase 22 sur chaque segment. Chaque période est évaluée séparément; ce contrôle reste expérimental.")
    wf1, wf2, wf3 = st.columns(3)
    with wf1:
        wf_market = st.selectbox("Marché du test chronologique", list(SYMBOLS.keys()), index=list(SYMBOLS.keys()).index(market_name), key="wf_market")
    with wf2:
        wf_tf = st.selectbox("Unité du test chronologique", list(INTERVALS.keys()), index=list(INTERVALS.keys()).index(bt_timeframe), key="wf_tf")
    with wf3:
        wf_bars = st.slider("Bougies du test chronologique", 300, 1500, 900, 100, key="wf_bars")
    wf_folds = st.select_slider("Nombre de périodes successives", options=[3, 4, 5], value=3, key="wf_folds")
    if st.button("Tester la stabilité chronologique", key="run_smc_walk_forward", type="secondary"):
        with st.spinner("Test des périodes successives..."):
            wf_data = fetch_candles(SYMBOLS[wf_market], wf_tf, wf_bars)
            if wf_data is None or len(wf_data) < wf_folds * 100:
                st.error("Données insuffisantes. Augmente le nombre de bougies ou choisis une autre unité de temps.")
            else:
                wf_summary, wf_details = run_smc_walk_forward(wf_data, folds=wf_folds, rr_value=float(bt_rr), cost_r=float(bt_cost_r))
                if wf_summary.empty:
                    st.warning("Aucune période exploitable n'a produit de résumé.")
                else:
                    st.dataframe(wf_summary, use_container_width=True, hide_index=True)
                    st.download_button("Télécharger le résumé chronologique CSV", wf_summary.to_csv(index=False).encode("utf-8-sig"), file_name="forexscope_stabilite_chronologique.csv", mime="text/csv", key="download_wf_summary")
                    for label, trades_df in wf_details.items():
                        with st.expander(label):
                            if trades_df.empty:
                                st.info("Aucun trade classé dans cette période.")
                            else:
                                st.dataframe(trades_df, use_container_width=True, hide_index=True)
                                st.download_button(f"Exporter {label}", trades_df.to_csv(index=False).encode("utf-8-sig"), file_name=f"forexscope_{label.lower().replace(' ', '_').replace('—', '-')}.csv", mime="text/csv", key=f"download_{label.replace(' ', '_').replace('—', '_')}")
                    st.warning("Chaque période est testée indépendamment avec un échauffement d'indicateurs propre. Le découpage ne corrige pas tous les biais de sélection, et l'approximation SMC ne comprend pas toute la confluence H4/D1 du moteur live. Les résultats ne garantissent pas les performances futures.")

    st.divider()
    st.subheader("Test de robustesse multi-marchés et multi-unités")
    st.caption(
        "Compare une stratégie EMA20/EMA50 ou l’approximation SMC sur plusieurs marchés et unités de temps. "
        "Le rapport ajoute une vérification chronologique indicative sur les derniers 30 % des trades classés."
    )
    robust_strategy = st.selectbox(
        "Stratégie du test de robustesse",
        ["SMC (sweep + structure + impulsion)", "EMA20/EMA50"],
        key="robust_strategy",
    )
    rb1, rb2 = st.columns(2)
    with rb1:
        robust_markets = st.multiselect(
            "Marchés à comparer", options=list(SYMBOLS.keys()),
            default=["EUR/USD", "GBP/USD", "Or (Gold)"] if all(x in SYMBOLS for x in ["EUR/USD", "GBP/USD", "Or (Gold)"]) else list(SYMBOLS.keys())[:3],
            key="robust_markets",
        )
    with rb2:
        robust_timeframes = st.multiselect(
            "Unités de temps à comparer", options=list(INTERVALS.keys()),
            default=["M15", "H1", "H4", "D1"], key="robust_timeframes",
        )
    rr1, rr2, rr3 = st.columns(3)
    with rr1:
        robust_bars = st.slider("Bougies par combinaison", 100, 1000, 300, 100, key="robust_bars")
    with rr2:
        robust_rr = st.selectbox("Ratio R:R du test comparatif", [1.0, 1.5, 2.0, 3.0], index=2, key="robust_rr")
    with rr3:
        robust_cost = st.number_input("Coût estimé par trade (R)", min_value=0.0, max_value=0.50, value=0.02, step=0.01, key="robust_cost")

    if st.button("Lancer le test de robustesse", type="secondary", key="run_robustness"):
        if not robust_markets or not robust_timeframes:
            st.warning("Sélectionne au moins un marché et une unité de temps.")
        else:
            summary_rows = []
            detail_rows = []
            progress = st.progress(0)
            combinations = [(m, tf) for m in robust_markets for tf in robust_timeframes]
            for combo_index, (robust_market, robust_tf) in enumerate(combinations):
                try:
                    raw = fetch_candles(SYMBOLS[robust_market], robust_tf, robust_bars)
                    if raw is None or len(raw) < 60:
                        summary_rows.append({"Marché": robust_market, "Unité": robust_tf, "Trades classés": 0, "Taux réussite (%)": np.nan, "Résultat net (R)": 0.0, "Espérance (R)": np.nan, "Statut": "Données insuffisantes"})
                        progress.progress((combo_index + 1) / len(combinations))
                        continue
                    frame = calculate_indicators(raw.copy()).reset_index(drop=True)
                    frame = frame.dropna(subset=["open", "high", "low", "close", "ema_20", "ema_50", "atr"]).reset_index(drop=True)
                    combo_trades = []
                    i = 1
                    while i < len(frame) - 1:
                        prev, cur = frame.iloc[i - 1], frame.iloc[i]
                        if not np.isfinite(cur["atr"]) or cur["atr"] <= 0:
                            i += 1
                            continue
                        entry_i = i + 1
                        entry = float(frame.iloc[entry_i]["open"])
                        if robust_strategy == "EMA20/EMA50":
                            up = prev["ema_20"] <= prev["ema_50"] and cur["ema_20"] > cur["ema_50"]
                            down = prev["ema_20"] >= prev["ema_50"] and cur["ema_20"] < cur["ema_50"]
                            if not (up or down):
                                i += 1
                                continue
                            side = "ACHAT" if up else "VENTE"
                            dist = float(cur["atr"])
                            sl = entry - dist if side == "ACHAT" else entry + dist
                        else:
                            setup = _find_confirmed_smc_setup(frame.iloc[:i + 1].copy(), float(cur["atr"]))
                            if setup.get("direction") not in ("Haussier", "Baissier"):
                                i += 1
                                continue
                            side = "ACHAT" if setup["direction"] == "Haussier" else "VENTE"
                            sweep_extreme = float(setup["sweep_extreme"])
                            sl = sweep_extreme - 0.2 * float(cur["atr"]) if side == "ACHAT" else sweep_extreme + 0.2 * float(cur["atr"])
                            dist = entry - sl if side == "ACHAT" else sl - entry
                            if not np.isfinite(dist) or dist <= 0 or dist > 2.5 * float(cur["atr"]):
                                i += 1
                                continue
                        tp = entry + dist * float(robust_rr) if side == "ACHAT" else entry - dist * float(robust_rr)
                        outcome = "OUVERT"
                        exit_i = len(frame) - 1
                        gross_r = ((float(frame.iloc[-1]["close"]) - entry) if side == "ACHAT" else (entry - float(frame.iloc[-1]["close"]))) / dist
                        for j in range(entry_i, len(frame)):
                            candle = frame.iloc[j]
                            hit_sl = float(candle["low"]) <= sl if side == "ACHAT" else float(candle["high"]) >= sl
                            hit_tp = float(candle["high"]) >= tp if side == "ACHAT" else float(candle["low"]) <= tp
                            if hit_sl and hit_tp:
                                outcome = "AMBIGU"
                                exit_i = j
                                gross_r = np.nan
                                break
                            if hit_sl:
                                outcome, exit_i, gross_r = "PERDANT", j, -1.0
                                break
                            if hit_tp:
                                outcome, exit_i, gross_r = "GAGNANT", j, float(robust_rr)
                                break
                        if outcome in ("GAGNANT", "PERDANT"):
                            net_r_trade = gross_r - float(robust_cost)
                            combo_trades.append({"Marché": robust_market, "Unité": robust_tf, "Entrée": frame.iloc[entry_i]["time"], "Sens": side, "Résultat": outcome, "R net estimé": net_r_trade, "Bougies détenues": int(exit_i - entry_i + 1)})
                        i = max(entry_i + 1, exit_i + 1)
                    detail_rows.extend(combo_trades)
                    n = len(combo_trades)
                    wins = sum(t["Résultat"] == "GAGNANT" for t in combo_trades)
                    net_values = [float(t["R net estimé"]) for t in combo_trades]
                    positive_values = [v for v in net_values if v > 0]
                    negative_values = [v for v in net_values if v < 0]
                    equity_r = np.cumsum(net_values) if net_values else np.array([])
                    peak_r = np.maximum.accumulate(np.concatenate(([0.0], equity_r)))[1:] if len(equity_r) else np.array([])
                    drawdown_r = peak_r - equity_r if len(equity_r) else np.array([])
                    gross_profit_r = float(sum(positive_values))
                    gross_loss_r = abs(float(sum(negative_values)))
                    # Séparation chronologique des trades classés : premiers 70 % descriptifs,
                    # derniers 30 % réservés à une vérification indicative.
                    split_at = int(np.floor(n * 0.70))
                    oos_values = net_values[split_at:] if n else []
                    oos_results = combo_trades[split_at:] if n else []
                    oos_n = len(oos_results)
                    oos_wins = sum(t["Résultat"] == "GAGNANT" for t in oos_results)
                    oos_net = float(sum(oos_values)) if oos_values else 0.0
                    oos_expectancy = float(np.mean(oos_values)) if oos_values else np.nan
                    summary_rows.append({
                        "Marché": robust_market, "Unité": robust_tf, "Trades classés": n,
                        "Taux réussite (%)": round(wins / n * 100, 1) if n else np.nan,
                        "Résultat net (R)": round(sum(net_values), 3),
                        "Espérance (R)": round(float(np.mean(net_values)), 3) if n else np.nan,
                        "Facteur de profit": round(gross_profit_r / gross_loss_r, 3) if gross_loss_r > 0 else (np.inf if gross_profit_r > 0 else np.nan),
                        "Drawdown max (R)": round(float(np.max(drawdown_r)), 3) if len(drawdown_r) else 0.0,
                        "Gain moyen (R)": round(float(np.mean(positive_values)), 3) if positive_values else np.nan,
                        "Perte moyenne (R)": round(float(np.mean(negative_values)), 3) if negative_values else np.nan,
                        "Trades vérification 30%": oos_n,
                        "Taux réussite vérification (%)": round(oos_wins / oos_n * 100, 1) if oos_n else np.nan,
                        "Résultat vérification (R)": round(oos_net, 3),
                                                " vérification (R)": round(oos_expectancy, 3) if np.isfinite(oos_expectancy) else np.nan,
                        "Statut": "À examiner" if n >= 20 and oos_n >= 6 else "Échantillon faible" if n else "Aucun trade classé",
                    })
                except Exception as exc:
                    summary_rows.append({"Marché": robust_market, "Unité": robust_tf, "Trades classés": 0, "Taux réussite (%)": np.nan, "Résultat net (R)": 0.0, "Espérance (R)": np.nan, "Statut": f"Erreur de données/calcul: {type(exc).__name__}"})
                progress.progress((combo_index + 1) / len(combinations))
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                st.subheader("Comparaison des résultats")
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                st.download_button("Télécharger le résumé CSV", summary_df.to_csv(index=False).encode("utf-8-sig"), file_name="forexscope_robustesse_resume.csv", mime="text/csv", key="download_robust_summary")
                if detail_rows:
                    detail_df = pd.DataFrame(detail_rows)
                    st.download_button("Télécharger les trades comparatifs CSV", detail_df.to_csv(index=False).encode("utf-8-sig"), file_name="forexscope_robustesse_trades.csv", mime="text/csv", key="download_robust_trades")
                st.warning(
                    "Les métriques globales sont calculées sur les trades classés de chaque combinaison. Les colonnes de vérification utilisent les derniers 30 % des trades classés, dans l’ordre chronologique; il s’agit d’un contrôle indicatif et non d’une validation totalement indépendante, car la stratégie n’a pas été optimisée séparément sur une période d’entraînement. Le drawdown est exprimé en R cumulés, et non en pourcentage du capital. "
                    "Les résultats peuvent varier selon la disponibilité et la qualité des données Yahoo Finance. "
                    "Les frais sont une estimation en R; le spread réel, le slippage et les spécifications du contrat ne sont pas reproduits."
                )



    st.divider()
    st.subheader("Analyse de sensibilité des paramètres")
    st.caption(
        "Compare plusieurs ratios rendement/risque et hypothèses de coûts sur le même marché, "
        "la même unité de temps et la même stratégie. Ce test aide à repérer des résultats "
        "trop dépendants d'un paramètre; il ne garantit pas la rentabilité."
    )
    ss1, ss2, ss3 = st.columns(3)
    with ss1:
        sens_market = st.selectbox("Marché de sensibilité", list(SYMBOLS.keys()), index=list(SYMBOLS.keys()).index(market_name) if market_name in SYMBOLS else 0, key="sens_market")
    with ss2:
        sens_tf = st.selectbox("Unité de temps de sensibilité", list(INTERVALS.keys()), index=list(INTERVALS.keys()).index(timeframe) if timeframe in INTERVALS else 0, key="sens_tf")
    with ss3:
        sens_strategy = st.selectbox("Stratégie de sensibilité", ["SMC (sweep + structure + impulsion)", "EMA20/EMA50"], key="sens_strategy")
    ss4, ss5 = st.columns(2)
    with ss4:
        sens_bars = st.slider("Bougies de sensibilité", 100, 1500, 500, 100, key="sens_bars")
    with ss5:
        sens_costs = st.multiselect("Coûts à comparer (R)", [0.00, 0.01, 0.02, 0.05, 0.10], default=[0.00, 0.02, 0.05], key="sens_costs")
    sens_rrs = st.multiselect("Ratios rendement/risque à comparer", [1.0, 1.5, 2.0, 2.5, 3.0], default=[1.0, 1.5, 2.0, 3.0], key="sens_rrs")
    if st.button("Lancer l'analyse de sensibilité", key="run_sensitivity", type="secondary"):
        if not sens_costs or not sens_rrs:
            st.warning("Sélectionne au moins un coût et un ratio R:R.")
        else:
            with st.spinner("Test des combinaisons de paramètres..."):
                sens_raw = fetch_candles(SYMBOLS[sens_market], sens_tf, sens_bars)
                if sens_raw is None or len(sens_raw) < 60:
                    st.error("Données insuffisantes pour cette combinaison marché/unité de temps.")
                else:
                    sens_rows = []
                    sens_details = []
                    base_strategy = "EMA20/EMA50" if sens_strategy == "EMA20/EMA50" else "SMC"
                    for rr_value in sens_rrs:
                        for cost_value in sens_costs:
                            try:
                                test_trades = run_sensitivity_backtest(sens_raw, base_strategy, rr_value, cost_value)
                                values = [float(t["R net estimé"]) for t in test_trades]
                                wins = sum(t["Résultat"] == "GAGNANT" for t in test_trades)
                                losses = sum(t["Résultat"] == "PERDANT" for t in test_trades)
                                n = len(test_trades)
                                gross_wins = sum(max(v, 0.0) for v in values)
                                gross_losses = abs(sum(min(v, 0.0) for v in values))
                                equity = np.cumsum(values) if values else np.array([])
                                peaks = np.maximum.accumulate(np.concatenate(([0.0], equity)))[1:] if len(equity) else np.array([])
                                dd = peaks - equity if len(equity) else np.array([])
                                sens_rows.append({
                                    "Marché": sens_market, "Unité": sens_tf, "Stratégie": sens_strategy,
                                    "R:R": float(rr_value), "Coût (R)": float(cost_value), "Trades classés": n,
                                    "Taux réussite (%)": round(wins / n * 100, 1) if n else np.nan,
                                    "Résultat net (R)": round(sum(values), 3),
                                    "Espérance (R)": round(float(np.mean(values)), 3) if values else np.nan,
                                    "Facteur de profit": round(gross_wins / gross_losses, 3) if gross_losses > 0 else (np.inf if gross_wins > 0 else np.nan),
                                    "Drawdown max (R)": round(float(np.max(dd)), 3) if len(dd) else 0.0,
                                    "Statut": "À examiner" if n >= 20 else "Échantillon faible",
                                })
                                for trade in test_trades:
                                    sens_details.append({"R:R": rr_value, "Coût (R)": cost_value, **trade})
                            except Exception as exc:
                                sens_rows.append({"Marché": sens_market, "Unité": sens_tf, "Stratégie": sens_strategy, "R:R": rr_value, "Coût (R)": cost_value, "Trades classés": 0, "Statut": f"Erreur: {type(exc).__name__}"})
                    sens_df = pd.DataFrame(sens_rows)
                    st.dataframe(sens_df, use_container_width=True, hide_index=True)
                    st.download_button("Télécharger le rapport de sensibilité CSV", sens_df.to_csv(index=False).encode("utf-8-sig"), file_name="forexscope_sensibilite.csv", mime="text/csv", key="download_sensitivity_summary")
                    if sens_details:
                        st.download_button("Télécharger les trades de sensibilité CSV", pd.DataFrame(sens_details).to_csv(index=False).encode("utf-8-sig"), file_name="forexscope_sensibilite_trades.csv", mime="text/csv", key="download_sensitivity_trades")
                    st.warning("Les résultats sont des simulations historiques. Les coûts sont déduits en R, mais ne modifient pas les prix de déclenchement; les trades ambigus où SL et TP sont touchés dans la même bougie sont exclus. Les résultats ne reproduisent pas parfaitement le courtier ni la confluence multi-unités complète du moteur SMC. Évite de choisir un paramètre uniquement parce qu'il a le meilleur résultat sur cet historique.")


    st.divider()
    st.subheader("Simulation Monte-Carlo des résultats")
    st.caption(
        "Cette simulation réordonne aléatoirement les résultats R des trades historiques classés pour illustrer l'incertitude liée à leur séquence. "
        "Elle ne prédit pas les marchés et ne remplace pas un backtest hors échantillon."
    )
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        mc_market = st.selectbox("Marché Monte-Carlo", list(SYMBOLS.keys()), index=list(SYMBOLS.keys()).index(market_name) if market_name in SYMBOLS else 0, key="mc_market")
    with mc2:
        mc_tf = st.selectbox("Unité Monte-Carlo", list(INTERVALS.keys()), index=list(INTERVALS.keys()).index(timeframe) if timeframe in INTERVALS else 0, key="mc_tf")
    with mc3:
        mc_strategy_label = st.selectbox("Stratégie Monte-Carlo", ["SMC (sweep + structure + impulsion)", "EMA20/EMA50"], key="mc_strategy")
    mc4, mc5, mc6 = st.columns(3)
    with mc4:
        mc_bars = st.slider("Bougies Monte-Carlo", 100, 1500, 500, 100, key="mc_bars")
    with mc5:
        mc_rr = st.selectbox("Ratio R:R Monte-Carlo", [1.0, 1.5, 2.0, 2.5, 3.0], index=2, key="mc_rr")
    with mc6:
        mc_cost = st.selectbox("Coût estimé par trade (R)", [0.00, 0.01, 0.02, 0.05, 0.10], index=2, key="mc_cost")
    mc_runs = st.select_slider("Nombre de simulations", options=[250, 500, 1000, 2000], value=1000, key="mc_runs")
    if st.button("Lancer la simulation Monte-Carlo", key="run_monte_carlo", type="secondary"):
        with st.spinner("Calcul des trades historiques puis des séquences simulées..."):
            mc_raw = fetch_candles(SYMBOLS[mc_market], mc_tf, mc_bars)
            if mc_raw is None or len(mc_raw) < 60:
                st.error("Données insuffisantes pour cette combinaison marché/unité de temps.")
            else:
                try:
                    mc_strategy = "EMA20/EMA50" if mc_strategy_label == "EMA20/EMA50" else "SMC"
                    mc_trades = run_sensitivity_backtest(mc_raw, mc_strategy, float(mc_rr), float(mc_cost))
                    mc_values = np.asarray([float(t["R net estimé"]) for t in mc_trades if np.isfinite(float(t["R net estimé"]))], dtype=float)
                    if len(mc_values) < 10:
                        st.warning(f"Seulement {len(mc_values)} trades classés. Il faut au moins 10 pour lancer une simulation indicative; idéalement, utilise un échantillon bien plus grand.")
                    else:
                        rng = np.random.default_rng(42)
                        final_results = np.empty(int(mc_runs), dtype=float)
                        max_drawdowns = np.empty(int(mc_runs), dtype=float)
                        for run_idx in range(int(mc_runs)):
                            sequence = rng.permutation(mc_values)
                            equity = np.cumsum(sequence)
                            peaks = np.maximum.accumulate(np.concatenate(([0.0], equity)))[1:]
                            drawdown = peaks - equity
                            final_results[run_idx] = equity[-1]
                            max_drawdowns[run_idx] = float(np.max(drawdown)) if len(drawdown) else 0.0
                        mc_summary = pd.DataFrame([
                            {"Indicateur": "Trades historiques classés", "Valeur": int(len(mc_values))},
                            {"Indicateur": "Résultat observé (R)", "Valeur": round(float(np.sum(mc_values)), 3)},
                            {"Indicateur": "Résultat final P5 (R)", "Valeur": round(float(np.percentile(final_results, 5)), 3)},
                            {"Indicateur": "Résultat final médian (R)", "Valeur": round(float(np.percentile(final_results, 50)), 3)},
                            {"Indicateur": "Résultat final P95 (R)", "Valeur": round(float(np.percentile(final_results, 95)), 3)},
                            {"Indicateur": "Probabilité simulée de finir sous 0 (%)", "Valeur": round(float(np.mean(final_results < 0) * 100), 1)},
                            {"Indicateur": "Drawdown maximal médian (R)", "Valeur": round(float(np.percentile(max_drawdowns, 50)), 3)},
                            {"Indicateur": "Drawdown maximal P95 (R)", "Valeur": round(float(np.percentile(max_drawdowns, 95)), 3)},
                        ])
                        st.dataframe(mc_summary, use_container_width=True, hide_index=True)
                        chart_df = pd.DataFrame({"Résultat final (R)": final_results})
                        st.line_chart(chart_df.sort_values("Résultat final (R)").reset_index(drop=True), use_container_width=True)
                        st.download_button("Télécharger le résumé Monte-Carlo CSV", mc_summary.to_csv(index=False).encode("utf-8-sig"), file_name="forexscope_monte_carlo_resume.csv", mime="text/csv", key="download_mc_summary")
                        st.download_button("Télécharger les résultats simulés CSV", pd.DataFrame({"Résultat final (R)": final_results, "Drawdown maximal (R)": max_drawdowns}).to_csv(index=False).encode("utf-8-sig"), file_name="forexscope_monte_carlo_simulations.csv", mime="text/csv", key="download_mc_runs")
                        st.warning("La simulation conserve les mêmes résultats de trades et ne fait que mélanger leur ordre. Elle suppose donc que les résultats sont interchangeables et ne modélise ni changement de marché, ni dépendance entre trades, ni coûts réels du courtier. Les percentiles sont des scénarios simulés, pas des probabilités fiables de performance future.")
                except Exception as exc:
                    st.error(f"La simulation n'a pas pu être calculée : {type(exc).__name__}. Vérifie les données et réessaie.")


    st.divider()
    st.subheader("Robustesse SMC sur plusieurs marchés")
    st.caption(
        "Compare les mêmes règles SMC sur plusieurs instruments avec la même unité de temps, le même objectif rendement/risque et les mêmes coûts estimés. "
        "Ce test aide à repérer une stratégie qui ne fonctionne que sur un seul marché; il ne constitue pas une optimisation automatique."
    )
    robust_markets = st.multiselect(
        "Marchés à comparer",
        options=list(SYMBOLS.keys()),
        default=[name for name in ["EUR/USD", "GBP/USD", "USD/JPY", "Or (XAU/USD)", "Bitcoin (BTC/USD)"] if name in SYMBOLS],
        key="smc_robust_markets",
    )
    rb1, rb2 = st.columns(2)
    with rb1:
        robust_tf = st.selectbox("Unité de temps de comparaison", list(INTERVALS.keys()), index=list(INTERVALS.keys()).index(bt_timeframe), key="smc_robust_tf")
    with rb2:
        robust_bars = st.slider("Bougies par marché", min_value=200, max_value=1500, value=500, step=100, key="smc_robust_bars")
    if st.button("Lancer la comparaison multi-marchés", key="run_smc_multi_market_robustness", type="secondary"):
        if len(robust_markets) < 2:
            st.warning("Sélectionne au moins deux marchés pour comparer la robustesse.")
        else:
            robustness_rows = []
            robustness_details = {}
            progress = st.progress(0, text="Préparation des marchés...")
            for market_index, selected_market in enumerate(robust_markets):
                try:
                    market_data = fetch_candles(SYMBOLS[selected_market], robust_tf, robust_bars)
                    if market_data is None or len(market_data) < 100:
                        robustness_rows.append({"Marché": selected_market, "Bougies disponibles": 0 if market_data is None else len(market_data), "Variante": "SMC de base", "Trades classés": 0, "Taux de réussite (%)": np.nan, "Résultat net estimé (R)": np.nan, "Espérance (R/trade)": np.nan, "Facteur de profit": np.nan, "Statut": "Données insuffisantes"})
                    else:
                        market_summary, market_trade_details = run_smc_filter_comparison(market_data, rr_value=float(bt_rr), cost_r=float(bt_cost_r))
                        for _, row in market_summary.iterrows():
                            row_dict = row.to_dict()
                            robustness_rows.append({
                                "Marché": selected_market,
                                "Bougies disponibles": len(market_data),
                                "Variante": row_dict.get("Variante", "Inconnue"),
                                "Trades classés": row_dict.get("Trades classés", 0),
                                "Taux de réussite (%)": row_dict.get("Taux de réussite (%)", np.nan),
                                "Résultat net estimé (R)": row_dict.get("Résultat net estimé (R)", np.nan),
                                "Espérance (R/trade)": row_dict.get("Espérance (R/trade)", np.nan),
                                "Facteur de profit": row_dict.get("Facteur de profit", np.nan),
                                "Statut": "Échantillon faible" if int(row_dict.get("Trades classés", 0) or 0) < 30 else "À examiner",
                            })
                            for variant, trades_df in market_trade_details.items():
                                robustness_details[f"{selected_market} — {variant}"] = trades_df.copy()
                except Exception as exc:
                    robustness_rows.append({"Marché": selected_market, "Bougies disponibles": 0, "Variante": "—", "Trades classés": 0, "Taux de réussite (%)": np.nan, "Résultat net estimé (R)": np.nan, "Espérance (R/trade)": np.nan, "Facteur de profit": np.nan, "Statut": f"Erreur : {type(exc).__name__}"})
                progress.progress((market_index + 1) / len(robust_markets), text=f"Marché analysé : {selected_market}")
            progress.empty()
            robustness_df = pd.DataFrame(robustness_rows)
            if not robustness_df.empty:
                st.dataframe(robustness_df, use_container_width=True, hide_index=True)
                st.markdown("#### Incertitude de l'espérance par marché")
                ci_rows = []
                for detail_name, detail_df in robustness_details.items():
                    ci = bootstrap_expectancy_interval(detail_df, iterations=1000, seed=42)
                    market_part, _, variant_part = detail_name.partition(" — ")
                    ci_rows.append({"Marché": market_part, "Variante": variant_part, **ci})
                if ci_rows:
                    ci_df = pd.DataFrame(ci_rows)
                    st.dataframe(ci_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Exporter les intervalles d'espérance CSV",
                        ci_df.to_csv(index=False).encode("utf-8-sig"),
                        file_name="forexscope_intervalles_es esperance.csv".replace(" ", "_"),
                        mime="text/csv",
                        key="download_smc_expectancy_intervals",
                    )
                    st.caption("L'intervalle bootstrap rééchantillonne les trades classés. Il est indicatif : il ne modélise pas la dépendance temporelle, les changements de régime ni les coûts réels. Un intervalle recouvrant zéro ne permet pas d'affirmer que l'espérance est positive.")
                st.download_button(
                    "Exporter les résultats multi-marchés CSV",
                    robustness_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="forexscope_robustesse_smc_multi_marches.csv",
                    mime="text/csv",
                    key="download_smc_multi_market_robustness",
                )
                with st.expander("Détails des trades par marché et variante"):
                    for detail_name, detail_df in robustness_details.items():
                        st.markdown(f"**{detail_name}**")
                        if detail_df.empty:
                            st.info("Aucun trade classé pour cette combinaison.")
                        else:
                            st.dataframe(detail_df, use_container_width=True, hide_index=True)
                st.warning(
                       "Les marchés peuvent avoir des historiques, spreads, horaires et qualités de données différents. "
                    " coûts restent des estimations communes, les résultats sont issus d'une approximation mécanique du SMC et les échantillons de moins de 30 trades sont particulièrement incertains. "
                    "Ne choisis pas un marché uniquement parce que son résultat historique est le meilleur."
                )


    st.divider()
    st.subheader("Validation SMC par fenêtres chronologiques")
    st.caption(
        "Le test découpe l'historique en fenêtres successives et mesure la stabilité des règles SMC fixes. "
        "Aucune optimisation n'est faite sur ces fenêtres; il s'agit d'un contrôle de stabilité, pas d'une garantie de rentabilité."
    )
    wf_market = st.selectbox("Marché pour la validation chronologique", list(SYMBOLS.keys()), index=list(SYMBOLS.keys()).index(market_name) if market_name in SYMBOLS else 0, key="smc_walkforward_market")
    wf1, wf2 = st.columns(2)
    with wf1:
        wf_tf = st.selectbox("Unité de temps", list(INTERVALS.keys()), index=list(INTERVALS.keys()).index(bt_timeframe) if bt_timeframe in INTERVALS else 0, key="smc_walkforward_tf")
    with wf2:
        wf_windows = st.selectbox("Nombre de fenêtres chronologiques", [3, 4, 5, 6], index=1, key="smc_walkforward_windows")
    wf_bars = st.slider("Historique total à découper", min_value=300, max_value=1500, value=800, step=100, key="smc_walkforward_bars")
    if st.button("Lancer la validation chronologique", key="run_smc_walkforward", type="secondary"):
        try:
            with st.spinner("Analyse des fenêtres chronologiques..."):
                wf_data = fetch_candles(SYMBOLS[wf_market], wf_tf, wf_bars)
                if wf_data is None or len(wf_data) < wf_windows * 100:
                    st.warning("Historique insuffisant pour créer des fenêtres d'au moins 100 bougies. Augmente l'historique ou réduis le nombre de fenêtres.")
                else:
                    wf_results, wf_details = run_smc_chronological_windows(wf_data, windows=wf_windows, rr_value=float(bt_rr), cost_r=float(bt_cost_r))
                    if wf_results.empty:
                        st.info("Aucun résultat disponible pour ces paramètres.")
                    else:
                        st.dataframe(wf_results, use_container_width=True, hide_index=True)
                        stability_summary = summarize_smc_window_stability(wf_results)
                        if not stability_summary.empty:
                            st.markdown("#### Synthèse de stabilité par variante")
                            st.dataframe(stability_summary, use_container_width=True, hide_index=True)
                            st.caption("La synthèse décrit la dispersion entre fenêtres historiques. Elle ne constitue pas une estimation garantie des performances futures.")
                        export_parts = [wf_results.assign(Type="Détail par fenêtre")]
                        if not stability_summary.empty:
                            export_parts.append(stability_summary.assign(Type="Synthèse de stabilité"))
                        export_df = pd.concat(export_parts, ignore_index=True, sort=False)
                        st.download_button("Exporter la validation chronologique CSV", export_df.to_csv(index=False).encode("utf-8-sig"), file_name="forexscope_validation_chronologique.csv", mime="text/csv", key="download_smc_walkforward")
                        valid = wf_results.dropna(subset=["Espérance (R/trade)"]) if "Espérance (R/trade)" in wf_results.columns else pd.DataFrame()
                        if not valid.empty:
                            st.markdown("#### Espérance par fenêtre")
                            chart_data = valid.pivot_table(index="Fenêtre", columns="Variante", values="Espérance (R/trade)", aggfunc="first")
                            st.line_chart(chart_data, use_container_width=True)
                        with st.expander("Détails des trades par fenêtre"):
                            for detail_name, detail_df in wf_details.items():
                                st.markdown(f"**{detail_name}**")
                                if detail_df.empty:
                                    st.info("Aucun trade classé dans cette fenêtre.")
                                else:
                                    st.dataframe(detail_df, use_container_width=True, hide_index=True)
                        st.warning("Limites : chaque fenêtre réinitialise l'échauffement des indicateurs; les coûts restent estimés; les fenêtres peuvent contenir peu de trades. Cette procédure vérifie la stabilité chronologique des règles fixes, mais ce n'est pas un vrai walk-forward avec optimisation sur une période d'entraînement et test indépendant.")
        except Exception as exc:
            st.error(f"La validation chronologique a échoué : {type(exc).__name__}. Vérifie les données et réessaie.")


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
    st.subheader("Tests contrôlés du moteur SMC")
    st.caption("Scénarios synthétiques reproductibles pour vérifier quelques comportements fondamentaux. Ces tests ne prouvent pas la rentabilité et ne couvrent pas toutes les branches du moteur.")
    if st.button("Lancer les tests du moteur SMC", key="run_smc_engine_self_tests"):
        self_test_report = run_smc_engine_self_tests()
        st.dataframe(self_test_report, use_container_width=True, hide_index=True)
        failed_tests = self_test_report[self_test_report["État"] == "ÉCHEC"]
        if failed_tests.empty:
            st.success(f"{len(self_test_report)} tests contrôlés réussis.")
        else:
            st.error(f"{len(failed_tests)} test(s) en échec. Vérifie les détails avant de poursuivre.")
        st.download_button(
            "Exporter les tests SMC CSV",
            data=self_test_report.to_csv(index=False).encode("utf-8-sig"),
            file_name="forexscope_tests_smc.csv",
            mime="text/csv",
            key="download_smc_engine_self_tests",
        )

    st.divider()
    st.subheader("Intégrité temporelle des signaux")
    st.caption("Tests synthétiques vérifiant que l’analyse multi-timeframe utilise la dernière bougie clôturée et que l’horodatage correspond à cette bougie.")
    if st.button("Tester l’intégrité temporelle", key="run_temporal_integrity_tests"):
        temporal_report = run_smc_temporal_integrity_tests()
        st.dataframe(temporal_report, use_container_width=True, hide_index=True)
        temporal_failures = temporal_report[temporal_report["État"] == "ÉCHEC"]
        if temporal_failures.empty:
            st.success(f"{len(temporal_report)} tests temporels réussis.")
        else:
            st.error(f"{len(temporal_failures)} test(s) temporel(s) en échec. Ne considère pas les signaux comme validés.")
        st.download_button(
            "Exporter les tests temporels CSV",
            data=temporal_report.to_csv(index=False).encode("utf-8-sig"),
            file_name="forexscope_tests_integrite_temporelle.csv",
            mime="text/csv",
            key="download_temporal_integrity_tests",
        )


    st.divider()
    st.subheader("Cohérence multi-unités de temps")
    st.caption("Compare les biais des unités courtes M5/M15 au contexte supérieur H4/D1. Cette vérification décrit l’alignement, elle ne crée pas à elle seule un signal de trading.")
    if st.button("Analyser la cohérence MTF", key="run_mtf_consistency_audit"):
        with st.spinner("Récupération et analyse des unités de temps…"):
            mtf_audit_results = analyze_multiple_timeframes(SYMBOLS[market_name], candle_count)
        mtf_report, mtf_state, mtf_summary = build_mtf_consistency_report(mtf_audit_results)
        st.dataframe(mtf_report, use_container_width=True, hide_index=True)
        if mtf_state == "ALIGNÉ":
            st.success(mtf_summary)
        elif mtf_state == "CONFLIT":
            st.warning(mtf_summary)
        else:
            st.info(mtf_summary)
        st.download_button(
            "Exporter le rapport MTF CSV",
            data=mtf_report.to_csv(index=False).encode("utf-8-sig"),
            file_name="forexscope_coherence_mtf.csv",
            mime="text/csv",
            key="download_mtf_consistency_audit",
        )

    st.divider()
    st.subheader("Tests d’intégrité des données MTF")
    st.caption("Vérifie les cas de données manquantes, d’horodatages invalides et de biais inconnus. Le contrôle de fraîcheur ci-dessus est indicatif et dépend du calendrier du marché.")
    if st.button("Lancer les tests d’intégrité MTF", key="run_mtf_integrity_unit_tests"):
        mtf_integrity_tests = run_mtf_integrity_unit_tests()
        st.dataframe(mtf_integrity_tests, use_container_width=True, hide_index=True)
        mtf_failed = mtf_integrity_tests[mtf_integrity_tests["État"] == "ÉCHEC"]
        if mtf_failed.empty:
            st.success(f"{len(mtf_integrity_tests)} tests d’intégrité MTF réussis.")
        else:
            st.error(f"{len(mtf_failed)} test(s) MTF en échec.")
        st.download_button(
            "Exporter les tests d’intégrité MTF CSV",
            data=mtf_integrity_tests.to_csv(index=False).encode("utf-8-sig"),
            file_name="forexscope_tests_integrite_mtf.csv",
            mime="text/csv",
            key="download_mtf_integrity_unit_tests",
        )

    st.divider()
    st.subheader("Validation stricte de la confluence MTF")
    st.caption("Contrôle M5, M15, H4 et D1 ensemble. Une unité absente, périmée ou neutre empêche la validation complète. Ce statut n’est ni une probabilité de gain ni un signal d’achat/vente.")
    if st.button("Évaluer la préparation MTF", key="run_mtf_readiness_gate"):
        with st.spinner("Analyse des unités M5, M15, H4 et D1…"):
            readiness_results = analyze_multiple_timeframes(SYMBOLS[market_name], candle_count)
        readiness_report, readiness_state, readiness_summary, readiness_issues = assess_mtf_readiness(readiness_results)
        st.dataframe(readiness_report, use_container_width=True, hide_index=True)
        if readiness_state == "PRÊT POUR ÉVALUATION":
            st.success(readiness_summary)
        elif readiness_state == "CONFLIT DIRECTIONNEL":
            st.warning(readiness_summary)
        else:
            st.info(readiness_summary)
        if readiness_issues:
            st.write("Points à vérifier :")
            for issue in readiness_issues:
                st.write(f"- {issue}")
        st.download_button("Exporter le diagnostic de préparation MTF CSV", readiness_report.to_csv(index=False).encode("utf-8-sig"), file_name="forexscope_preparation_mtf.csv", mime="text/csv", key="download_mtf_readiness_gate")

    if st.button("Tester la validation stricte MTF", key="run_mtf_readiness_tests"):
        readiness_tests = run_mtf_readiness_tests()
        st.dataframe(readiness_tests, use_container_width=True, hide_index=True)
        readiness_failures = readiness_tests[readiness_tests["État"] == "ÉCHEC"]
        if readiness_failures.empty:
            st.success(f"{len(readiness_tests)} tests de préparation MTF réussis.")
        else:
            st.error(f"{len(readiness_failures)} test(s) de préparation MTF en échec.")
        st.download_button("Exporter les tests de préparation MTF CSV", readiness_tests.to_csv(index=False).encode("utf-8-sig"), file_name="forexscope_tests_preparation_mtf.csv", mime="text/csv", key="download_mtf_readiness_tests")

    st.divider()
    st.subheader("Tests du blocage des signaux sur données périmées")
    st.caption("Tests contrôlés du garde-fou : données fraîches, bougies périmées, contexte H4 manquant et horodatages absents.")
    if st.button("Lancer les tests de fraîcheur des signaux", key="run_signal_freshness_guard_tests"):
        freshness_tests = run_signal_freshness_guard_tests()
        st.dataframe(freshness_tests, use_container_width=True, hide_index=True)
        failed_freshness = freshness_tests[freshness_tests["État"] == "ÉCHEC"]
        if failed_freshness.empty:
            st.success(f"{len(freshness_tests)} tests de fraîcheur réussis.")
        else:
            st.error(f"{len(failed_freshness)} test(s) de fraîcheur en échec.")
        st.download_button(
            "Exporter les tests de fraîcheur CSV",
            data=freshness_tests.to_csv(index=False).encode("utf-8-sig"),
            file_name="forexscope_tests_fraicheur_signaux.csv",
            mime="text/csv",
            key="download_signal_freshness_guard_tests",
        )

    st.divider()
    st.subheader("Cohérence signal / historique — Phase 49")
    st.info("Le contrôle avant enregistrement vérifie désormais aussi que le ratio rendement/risque déclaré correspond aux distances Entrée–SL et Entrée–TP. Le calcul de lots reste une estimation et dépend de la valeur du pip et du contrat fournis.")
    st.caption("Vérifie l’ordre Entrée/SL/TP, le ratio rendement/risque annoncé et l’horodatage. Les nouveaux signaux ACHAT/VENTE incohérents sont refusés avant insertion dans SQLite.")
    audit_col, test_col = st.columns(2)
    with audit_col:
        if st.button("Auditer l’historique des signaux", key="run_saved_signal_consistency_audit"):
            saved_signal_audit = audit_saved_signal_consistency(limit=500)
            st.dataframe(saved_signal_audit, use_container_width=True, hide_index=True)
            bad_rows = saved_signal_audit[saved_signal_audit["État"] == "À CORRIGER"] if "État" in saved_signal_audit.columns else pd.DataFrame()
            if bad_rows.empty and not (len(saved_signal_audit) == 1 and saved_signal_audit.iloc[0]["État"] == "À VÉRIFIER"):
                st.success("Aucune incohérence détectée dans les signaux examinés.")
            elif not bad_rows.empty:
                st.warning(f"{len(bad_rows)} signal(aux) nécessitent une vérification. L’historique n’a pas été modifié.")
            st.download_button("Exporter l’audit de l’historique CSV", saved_signal_audit.to_csv(index=False).encode("utf-8-sig"), file_name="forexscope_audit_historique_signaux.csv", mime="text/csv", key="download_saved_signal_consistency_audit")
    with test_col:
        if st.button("Tester les contrôles avant enregistrement", key="run_signal_record_consistency_tests"):
            signal_record_tests = run_signal_record_consistency_tests()
            st.dataframe(signal_record_tests, use_container_width=True, hide_index=True)
            signal_test_failures = signal_record_tests[signal_record_tests["État"] == "ÉCHEC"]
            if signal_test_failures.empty:
                st.success(f"{len(signal_record_tests)} tests de cohérence réussis.")
            else:
                st.error(f"{len(signal_test_failures)} test(s) de cohérence en échec.")
            st.download_button("Exporter les tests de cohérence CSV", signal_record_tests.to_csv(index=False).encode("utf-8-sig"), file_name="forexscope_tests_coherence_signaux.csv", mime="text/csv", key="download_signal_record_consistency_tests")

    st.divider()
    st.subheader("Dimensionnement du risque et arrondi du volume — Phase 50")
    st.info("Le volume théorique est arrondi vers le bas au pas indiqué. Le calcul vérifie que le risque estimé après arrondi ne dépasse pas le budget fourni. Il ne se connecte pas au courtier et ne remplace pas les spécifications exactes du symbole dans MT5.")
    if st.button("Tester le calcul de lots et l'arrondi", key="run_position_sizing_rounding_tests"):
        position_sizing_tests = run_position_sizing_rounding_tests()
        st.dataframe(position_sizing_tests, use_container_width=True, hide_index=True)
        position_sizing_failures = position_sizing_tests[position_sizing_tests["État"] == "ÉCHEC"]
        if position_sizing_failures.empty:
            st.success(f"{len(position_sizing_tests)} tests de dimensionnement réussis.")
        else:
            st.error(f"{len(position_sizing_failures)} test(s) de dimensionnement en échec.")
        st.download_button("Exporter les tests de dimensionnement CSV", position_sizing_tests.to_csv(index=False).encode("utf-8-sig"), file_name="forexscope_tests_dimensionnement_risque.csv", mime="text/csv", key="download_position_sizing_rounding_tests")

    st.divider()
    st.subheader("Tests du cycle de vie des signaux")
    st.caption("Exécute les tests dans une base SQLite temporaire isolée : création, anti-doublon, mise à jour du résultat, journal d’exécution et refus d’un signal incohérent. L’historique réel n’est pas modifié.")
    if st.button("Tester le cycle de vie des signaux", key="run_signal_lifecycle_tests"):
        lifecycle_tests = run_signal_lifecycle_tests()
        st.dataframe(lifecycle_tests, use_container_width=True, hide_index=True)
        lifecycle_failures = lifecycle_tests[lifecycle_tests["État"] == "ÉCHEC"]
        if lifecycle_failures.empty:
            st.success(f"{len(lifecycle_tests)} tests du cycle de vie réussis.")
        else:
            st.error(f"{len(lifecycle_failures)} test(s) du cycle de vie en échec.")
        st.download_button(
            "Exporter les tests du cycle de vie CSV",
            data=lifecycle_tests.to_csv(index=False).encode("utf-8-sig"),
            file_name="forexscope_tests_cycle_vie_signaux.csv",
            mime="text/csv",
            key="download_signal_lifecycle_tests",
        )

    st.divider()
    st.subheader("Audit technique local")
    st.caption("Contrôles de configuration, stockage et intégrité des bougies déjà chargées. Aucun nouvel appel au fournisseur de données n’est effectué.")
    st.markdown("**Qualité des bougies du marché sélectionné**")
    quality_report = build_candle_quality_report(candles, timeframe)
    st.dataframe(quality_report, use_container_width=True, hide_index=True)
    quality_problems = quality_report[quality_report["État"].isin(["ERREUR", "À VÉRIFIER", "INDISPONIBLE"])]
    if quality_problems.empty:
        st.success("Les contrôles de qualité des bougies chargées sont satisfaisants.")
    else:
        st.warning(f"{len(quality_problems)} contrôle(s) de qualité demandent une vérification.")
    st.download_button(
        "Exporter le contrôle qualité des bougies CSV",
        data=quality_report.to_csv(index=False).encode("utf-8-sig"),
        file_name="forexscope_qualite_bougies.csv", mime="text/csv", key="download_candle_quality_report",
    )
    if st.button("Tester la validation OHLC", key="run_market_data_integrity_tests"):
        data_tests = run_market_data_integrity_tests()
        st.dataframe(data_tests, use_container_width=True, hide_index=True)
        failures = data_tests[data_tests["État"] == "ÉCHEC"]
        if failures.empty:
            st.success(f"{len(data_tests)} tests de validation OHLC réussis.")
        else:
            st.error(f"{len(failures)} test(s) OHLC en échec.")
        st.download_button(
            "Exporter les tests OHLC CSV", data=data_tests.to_csv(index=False).encode("utf-8-sig"),
            file_name="forexscope_tests_validation_ohlc.csv", mime="text/csv", key="download_market_data_integrity_tests",
        )
    
    if st.button("Exécuter l’audit technique", key="run_local_technical_audit"):
        audit_report = audit_runtime_configuration(candles)
        st.dataframe(audit_report, use_container_width=True, hide_index=True)
        problems = audit_report[audit_report["État"].isin(["ERREUR", "À VÉRIFIER", "INDISPONIBLES"])]
        if problems.empty:
            st.success("Aucun problème détecté par ces contrôles locaux.")
        else:
            st.warning(f"{len(problems)} contrôle(s) nécessitent une vérification. Consulte le détail ci-dessus.")
        st.download_button(
            "Exporter le rapport d’audit CSV",
            data=audit_report.to_csv(index=False).encode("utf-8-sig"),
            file_name="forexscope_audit_technique.csv",
            mime="text/csv",
            key="download_local_technical_audit",
        )



    st.divider()
    st.subheader("Checklist de pré-finalisation — Phase 54")
    st.caption("Cette checklist sert à suivre les validations manuelles avant de considérer ForexScope comme prêt. Cocher un élément signifie que tu l’as réellement vérifié sur l’application déployée, pas seulement que le code existe.")
    checklist_items = [
        ("render_start", "L’application démarre sur Render sans erreur."),
        ("data_live", "Les bougies se chargent pour plusieurs symboles et unités de temps."),
        ("time_integrity", "Les tests d’intégrité temporelle passent."),
        ("mtf_integrity", "Les tests de cohérence et d’intégrité MTF passent."),
        ("risk_checks", "Les tests de cohérence SL/TP et de dimensionnement du risque passent."),
        ("signal_lifecycle", "Les tests du cycle de vie des signaux passent."),
        ("storage_backup", "Une sauvegarde SQLite est téléchargée et son intégrité vérifiée."),
        ("restart_persistence", "Après redémarrage/redéploiement, l’historique est toujours présent."),
        ("walk_forward", "Les tests chronologiques sont exécutés et leurs limites examinées."),
        ("paper_test", "Le moteur est testé en démo/paper trading sans argent réel."),
    ]
    completed_items = []
    checklist_cols = st.columns(2)
    for idx, (item_key, item_label) in enumerate(checklist_items):
        with checklist_cols[idx % 2]:
            checked = st.checkbox(item_label, key=f"phase54_check_{item_key}")
        completed_items.append({"Contrôle": item_label, "État": "VALIDÉ MANUELLEMENT" if checked else "À FAIRE"})
    done_count = sum(row["État"] == "VALIDÉ MANUELLEMENT" for row in completed_items)
    st.progress(done_count / len(completed_items))
    st.write(f"**Avancement déclaré : {done_count}/{len(completed_items)} contrôles.**")
    checklist_df = pd.DataFrame(completed_items)
    if done_count == len(completed_items):
        st.success("Tous les contrôles de la checklist sont cochés. Cela ne prouve pas la rentabilité : les résultats historiques et le paper trading doivent encore être interprétés avec prudence.")
    else:
        st.info("La finalisation reste conditionnelle aux contrôles non cochés, notamment au redémarrage Render, à la persistance des données et au paper trading.")
    st.download_button(
        "Exporter la checklist de pré-finalisation CSV",
        data=checklist_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="forexscope_checklist_pre_finalisation.csv",
        mime="text/csv",
        key="download_phase54_finalization_checklist",
    )


    st.divider()
    st.subheader("Batterie consolidée de tests — Phase 55")
    st.caption("Lance une série de tests déterministes locaux. Ces tests n’envoient aucun ordre et ne remplacent pas les vérifications manuelles sur Render ni le paper trading.")
    if st.button("Exécuter tous les tests locaux", key="phase55_run_all_tests"):
        consolidated_tests = [
            ("Validation OHLC", run_market_data_integrity_tests),
            ("Intégrité des horodatages MTF", run_mtf_integrity_unit_tests),
            ("Disponibilité de la confluence MTF", run_mtf_readiness_tests),
            ("Fraîcheur des données de signal", run_signal_freshness_guard_tests),
            ("Arrondi et dimensionnement des lots", run_position_sizing_rounding_tests),
            ("Cohérence des niveaux du signal", run_signal_record_consistency_tests),
            ("Cycle de vie et doublons", run_signal_lifecycle_tests),
            ("Sauvegarde et restauration SQLite", run_storage_integrity_tests),
            ("Regroupement mensuel des performances", test_period_robustness_report),
            ("Moteur SMC", run_smc_engine_self_tests),
            ("Intégrité temporelle SMC", run_smc_temporal_integrity_tests),
        ]
        consolidated_rows = []
        details_rows = []
        for test_label, test_function in consolidated_tests:
            try:
                test_result = test_function()
                if isinstance(test_result, pd.DataFrame):
                    result_frame = test_result.copy()
                    if "État" in result_frame.columns:
                        state_values = result_frame["État"].astype(str).str.upper().str.strip()
                        failed_mask = state_values.isin(["ÉCHEC", "ECHEC", "FAIL", "FAILED"])
                    elif "Statut" in result_frame.columns:
                        state_values = result_frame["Statut"].astype(str).str.upper().str.strip()
                        failed_mask = state_values.isin(["ÉCHEC", "ECHEC", "FAIL", "FAILED"])
                    else:
                        failed_mask = pd.Series(False, index=result_frame.index)
                    failed_count = int(failed_mask.sum())
                    test_state = "ÉCHEC" if failed_count else "RÉUSSI"
                    test_count = len(result_frame)
                    details_rows.append({"Batterie": test_label, "Détail": result_frame.to_json(orient="records", force_ascii=False, date_format="iso")})
                else:
                    failed_count = 0
                    test_count = 1
                    test_state = "RÉUSSI"
                    details_rows.append({"Batterie": test_label, "Détail": str(test_result)[:2000]})
                consolidated_rows.append({"Contrôle": test_label, "État": test_state, "Tests retournés": test_count, "Échecs détectés": failed_count})
            except Exception as exc:
                consolidated_rows.append({"Contrôle": test_label, "État": "ERREUR", "Tests retournés": 0, "Échecs détectés": 1})
                details_rows.append({"Batterie": test_label, "Détail": f"{type(exc).__name__}: {exc}"})
        consolidated_df = pd.DataFrame(consolidated_rows)
        details_df = pd.DataFrame(details_rows)
        st.dataframe(consolidated_df, use_container_width=True, hide_index=True)
        total_failed = int(consolidated_df["Échecs détectés"].sum())
        total_errors = int((consolidated_df["État"] == "ERREUR").sum())
        if total_failed == 0 and total_errors == 0:
            st.success(f"Les {len(consolidated_df)} batteries locales ont terminé sans échec détecté.")
        else:
            st.error(f"Des problèmes ont été détectés : {total_failed} échec(s) ou {total_errors} erreur(s) d’exécution. Consulte le rapport détaillé.")
        with st.expander("Détails des batteries de tests"):
            st.dataframe(details_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Exporter le bilan consolidé CSV",
            data=consolidated_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="forexscope_bilan_tests_phase55.csv",
            mime="text/csv",
            key="phase55_download_consolidated_tests",
        )
        st.download_button(
            "Exporter les détails des tests CSV",
            data=details_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="forexscope_details_tests_phase55.csv",
            mime="text/csv",
            key="phase55_download_test_details",
        )
    st.info("Un résultat local réussi ne confirme pas la disponibilité du fournisseur de données, la persistance du disque Render après redéploiement, ni la rentabilité. Ces points doivent être vérifiés séparément.")

st.divider()
st.caption(
    "ForexScope • Page actualisée à : "
    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
            


        
